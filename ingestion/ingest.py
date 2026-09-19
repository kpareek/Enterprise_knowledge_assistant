"""
End-to-end ingestion pipeline, with delta support by default:
    walk directory -> hash each file -> compare to manifest ->
    only load/chunk/embed files that are new or changed -> upsert ->
    save updated manifest.

Files whose content hash hasn't changed since the last run are skipped
entirely -- no re-loading, no re-chunking, and critically, no repeat
embedding API calls (or repeat local-model inference in free mode). This
is what makes adding one new document to a large corpus fast and cheap
instead of re-embedding everything every time.

Run:
    python -m ingestion.ingest                     # delta ingest data/sample_docs
    python -m ingestion.ingest --dir /path/to/docs  # delta ingest a different directory
    python -m ingestion.ingest --reset              # wipe collection + manifest, ingest everything fresh
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from ingestion.loaders import load_document, LOADER_MAP
from ingestion.chunking import chunk_documents
from ingestion.manifest import file_hash, load_manifest, save_manifest, record_entry
from rag.vector_store import VectorStore


def _find_supported_files(directory: str) -> list[str]:
    """Same walk + extension filter as loaders.load_directory, but just
    returns paths -- doesn't load file contents yet. Ingesting from
    two different directories that happen to share a filename is not
    supported (both write to the same 'source' metadata value), since
    delete_by_source() matches on filename only."""
    paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in LOADER_MAP:
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def _manifest_key(filepath: str) -> str:
    """Manifest keys are stored relative to the project root (BASE_DIR),
    not as absolute paths -- config.RAW_DOCS_DIR is built from an absolute
    BASE_DIR, so file paths from _find_supported_files are absolute, but an
    absolute-path key would bake this machine's directory layout into a
    file that's checked into git and read on other machines/checkouts."""
    return os.path.relpath(filepath, config.BASE_DIR)


def run_ingestion(directory: str = config.RAW_DOCS_DIR, reset: bool = False) -> dict:
    t0 = time.time()
    store = VectorStore()

    if reset:
        print("[INFO] --reset: wiping existing collection and manifest, ingesting everything fresh.")
        store.reset()
        manifest = {}
    else:
        manifest = load_manifest(config.MANIFEST_PATH)
        print(f"[INFO] Loaded manifest with {len(manifest)} previously-ingested file(s).")

    current_files = _find_supported_files(directory)
    current_keys = {_manifest_key(f) for f in current_files}

    # Only consider a manifest entry "removed" if it lived inside the
    # directory we're currently scanning -- otherwise ingesting
    # data/sample_docs alone would wrongly delete chunks tracked from a
    # previously-ingested data/external_enterpriserag_bench, since that
    # directory's files are (correctly) absent from this run's file list.
    directory_abs = os.path.abspath(directory)
    manifest_files_in_scope = {
        key for key in manifest
        if os.path.abspath(os.path.join(config.BASE_DIR, key)).startswith(directory_abs + os.sep)
        or os.path.abspath(os.path.join(config.BASE_DIR, key)) == directory_abs
    }
    removed = manifest_files_in_scope - current_keys
    for key in removed:
        source_name = os.path.basename(key)
        print(f"[INFO] Removing chunks for deleted file: {source_name}")
        store.delete_by_source(source_name)
        del manifest[key]

    new_files, changed_files, skipped_files = [], [], []
    total_chunks_added = 0

    for filepath in current_files:
        key = _manifest_key(filepath)
        h = file_hash(filepath)
        prior = manifest.get(key)

        if prior is not None and prior.get("hash") == h:
            skipped_files.append(filepath)
            continue

        source_name = os.path.basename(filepath)
        if prior is not None:
            # Content changed -- clear old chunks before re-embedding,
            # otherwise the stale ones would linger alongside the new.
            store.delete_by_source(source_name)
            changed_files.append(filepath)
        else:
            new_files.append(filepath)

        try:
            records = load_document(filepath)
        except Exception as e:
            print(f"[WARN] Failed to load {filepath}: {e}")
            continue

        chunks = chunk_documents(records)
        if chunks:
            store.upsert_chunks(chunks)
            total_chunks_added += len(chunks)

        manifest[key] = {"hash": h, **record_entry(len(chunks))}
        print(f"[INFO] {'Re-embedded' if prior else 'Embedded'} {source_name} -> {len(chunks)} chunks")

    save_manifest(config.MANIFEST_PATH, manifest)

    elapsed = round(time.time() - t0, 2)
    print(f"\n[DONE] {len(new_files)} new, {len(changed_files)} changed, "
          f"{len(skipped_files)} unchanged (skipped), {len(removed)} removed.")
    print(f"[INFO] {total_chunks_added} chunks embedded this run, in {elapsed}s.")
    print(f"[INFO] Collection now has {store.count()} total chunks across {len(manifest)} documents.")

    return {
        "new": len(new_files),
        "changed": len(changed_files),
        "unchanged_skipped": len(skipped_files),
        "removed": len(removed),
        "chunks_embedded_this_run": total_chunks_added,
        "total_chunks_in_collection": store.count(),
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest enterprise documents into the vector store (delta by default).")
    parser.add_argument("--dir", default=config.RAW_DOCS_DIR, help="Directory of source documents")
    parser.add_argument("--reset", action="store_true", help="Wipe existing collection and manifest, then ingest everything fresh")
    args = parser.parse_args()
    run_ingestion(args.dir, args.reset)
