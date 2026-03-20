#!/usr/bin/env python3
"""
Sequential (method-group-by-group) evaluation of KG similarity methods.

Produces the same output columns as evaluate_all_datasets.py, but processes
one group of methods at a time — freeing memory between groups — to keep
CPU / RAM usage low on large datasets.

Method groups (each group shares one underlying model):
  1. KEA            — graph-only, no neural model
  2. TransE+RotatE  — shared GraphEmbeddingSimilarity per row
  3. WL Kernel      — graph-only
  4. AA-KEA         — graph-only
  5. SNEA-BERT      — all 11 alphas share ONE call to calculate_snea_sbert_similarity
  6. KEA-BERT       — BERT model
  7. Semantic-WL    — SBERT model

Usage:
  # Run all method groups on the wikipedia dataset
  python evaluate_sequential.py \\
      --input  Benchmarking_Pipeline/Data/wikipedia_entity_swap_400_KGs.csv \\
      --output Benchmarking_Pipeline/Results_All_Methods/wikipedia_entity_swap_400_KGs_results.csv

  # Skip groups already done (resume from checkpoint)
  python evaluate_sequential.py \\
      --input  Benchmarking_Pipeline/Data/wikipedia_entity_swap_400_KGs.csv \\
      --output Benchmarking_Pipeline/Results_All_Methods/wikipedia_entity_swap_400_KGs_results.csv \\
      --skip-groups KEA TransE+RotatE "WL Kernel" AA-KEA

  # Run only specific groups
  python evaluate_sequential.py \\
      --input  Benchmarking_Pipeline/Data/wikipedia_entity_swap_400_KGs.csv \\
      --output Benchmarking_Pipeline/Results_All_Methods/wikipedia_entity_swap_400_KGs_results.csv \\
      --only-groups SNEA-BERT KEA-BERT

  # Quick test with 10 rows
  python evaluate_sequential.py --input ... --output ... --limit 10
"""

import ast
import csv
import gc
import argparse
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Column definitions (must match evaluate_all_datasets.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

SNEA_BERT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SNEA_BERT_ALPHA_NAMES = {a: f'snea_bert_alpha_{a}' for a in SNEA_BERT_ALPHAS}
SNEA_BERT_ALPHA_NAMES[1.0] = 'snea_bert_alpha_1.0_SNEA_alone'

ALL_METHODS = [
    'kea_similarity',
    'transe_similarity',
    'rotate_similarity',
    'wl_kernel_similarity',
    'aa_kea_similarity',
    *[SNEA_BERT_ALPHA_NAMES[a] for a in SNEA_BERT_ALPHAS],
    'kea_bert_similarity',
    'semantic_wl_similarity',
]

INPUT_FIELDNAMES = ['id', 'paragraph_1', 'paragraph_2', 'kg_1', 'kg_2']

# Each group: (display_name, [method_columns]).
# Methods inside a group are computed together (they share one model or one
# computation call), so they are NOT split further.
METHOD_GROUPS = [
    ('KEA',           ['kea_similarity']),
    ('TransE+RotatE', ['transe_similarity', 'rotate_similarity']),
    ('WL Kernel',     ['wl_kernel_similarity']),
    ('AA-KEA',        ['aa_kea_similarity']),
    ('SNEA-BERT',     [SNEA_BERT_ALPHA_NAMES[a] for a in SNEA_BERT_ALPHAS]),
    ('KEA-BERT',      ['kea_bert_similarity']),
    ('Semantic-WL',   ['semantic_wl_similarity']),
]

GROUP_NAMES = [g[0] for g in METHOD_GROUPS]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_triples(kg_str):
    """Parse a stringified list-of-lists into valid (len==3) triples."""
    try:
        parsed = ast.literal_eval(kg_str) if kg_str and kg_str.strip() else []
    except Exception:
        return []
    return [t for t in parsed if isinstance(t, list) and len(t) == 3]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-group score functions
# (Methods imported lazily inside each function so that only the model needed
#  for the current group is loaded into memory at any given time.)
# ─────────────────────────────────────────────────────────────────────────────

def _score_kea(kg1, kg2) -> dict:
    from Methods import calculate_kea_similarity  # noqa: PLC0415
    result = calculate_kea_similarity(kg1, kg2)
    return {'kea_similarity': result[0] if isinstance(result, tuple) else result}


def _score_transe_rotate(kg1, kg2) -> dict:
    from Methods import GraphEmbeddingSimilarity  # noqa: PLC0415
    scores = {}
    emb = GraphEmbeddingSimilarity(embedding_dim=50)
    try:
        scores['transe_similarity'] = emb.calculate_transe_similarity(kg1, kg2)
    except Exception:
        scores['transe_similarity'] = None
    try:
        scores['rotate_similarity'] = emb.calculate_rotate_similarity(kg1, kg2)
    except Exception:
        scores['rotate_similarity'] = None
    del emb
    return scores


def _score_wl(kg1, kg2) -> dict:
    from Methods import calculate_wl_kernel_similarity  # noqa: PLC0415
    return {'wl_kernel_similarity': calculate_wl_kernel_similarity(kg1, kg2)}


def _score_aa_kea(kg1, kg2) -> dict:
    from Methods import calculate_aa_kea_similarity  # noqa: PLC0415
    return {'aa_kea_similarity': calculate_aa_kea_similarity(kg1, kg2)}


def _score_snea_bert(kg1, kg2) -> dict:
    """Compute all 11 alpha blends from ONE call to calculate_snea_sbert_similarity."""
    from Methods import calculate_snea_sbert_similarity  # noqa: PLC0415
    _, debug = calculate_snea_sbert_similarity(kg1, kg2)
    wl_score    = debug.get('wl_score')
    sbert_score = debug.get('sbert_score_clipped')
    scores = {}
    for a in SNEA_BERT_ALPHAS:
        col = SNEA_BERT_ALPHA_NAMES[a]
        if wl_score is not None and sbert_score is not None:
            scores[col] = float(a * wl_score + (1.0 - a) * sbert_score)
        else:
            scores[col] = debug.get('sbert_score', debug.get('blended_score'))
    return scores


def _score_kea_bert(kg1, kg2) -> dict:
    from Methods import calculate_kea_bert_similarity_score  # noqa: PLC0415
    return {'kea_bert_similarity': calculate_kea_bert_similarity_score(kg1, kg2)}


def _score_semantic_wl(kg1, kg2) -> dict:
    from Methods import calculate_semantic_wl_similarity_score  # noqa: PLC0415
    return {'semantic_wl_similarity': calculate_semantic_wl_similarity_score(kg1, kg2)}


# Map group name → score function
_GROUP_FN = {
    'KEA':           _score_kea,
    'TransE+RotatE': _score_transe_rotate,
    'WL Kernel':     _score_wl,
    'AA-KEA':        _score_aa_kea,
    'SNEA-BERT':     _score_snea_bert,
    'KEA-BERT':      _score_kea_bert,
    'Semantic-WL':   _score_semantic_wl,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────

def process_sequential(
    input_file: str,
    output_file: str,
    limit: int | None = None,
    skip_groups: list[str] | None = None,
    only_groups: list[str] | None = None,
) -> None:
    """
    Process `input_file` one method-group at a time.

    After each group completes:
      • gc.collect() frees the group's model allocations
      • a checkpoint CSV is written so progress is not lost if a later group crashes

    Args:
        input_file:   path to input CSV (id, paragraph_1, paragraph_2, kg_1, kg_2)
        output_file:  path for the output / checkpoint CSV
        limit:        process only the first N rows (for testing)
        skip_groups:  group names to skip (e.g. already done in a prior run)
        only_groups:  if set, run only these group names (overrides skip_groups)
    """
    skip_groups = set(skip_groups or [])
    only_groups = set(only_groups) if only_groups else None

    # Decide which groups to run
    groups_to_run = [
        (name, cols) for name, cols in METHOD_GROUPS
        if (only_groups is None or name in only_groups)
        and name not in skip_groups
    ]

    if not groups_to_run:
        print("No groups selected — nothing to do.")
        return

    print(f"Groups to run : {[g[0] for g in groups_to_run]}")
    print(f"Input         : {input_file}")
    print(f"Output        : {output_file}\n")

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))
    if limit:
        all_rows = all_rows[:limit]
    total = len(all_rows)
    print(f"Rows to process: {total}\n")

    # If a checkpoint already exists, load it so we can merge new columns in
    out_path = Path(output_file)
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            existing = {row['id']: row for row in csv.DictReader(f)}
        print(f"  Loaded checkpoint: {len(existing)} rows from {output_file}")
    else:
        existing = {}

    # Build working results list: start from existing checkpoint or fresh base row
    results: list[dict] = []
    for row in all_rows:
        row_id = row.get('id', '')
        if row_id in existing:
            results.append(dict(existing[row_id]))
        else:
            results.append({k: row.get(k, '') for k in INPUT_FIELDNAMES})

    # ── Process each group ────────────────────────────────────────────────────
    all_done_cols: list[str] = []
    # Collect which columns are already in the checkpoint
    if existing:
        sample = next(iter(existing.values()))
        all_done_cols = [c for c in ALL_METHODS if c in sample and sample[c] not in ('', None)]

    score_fn_map = dict(_GROUP_FN)

    for group_name, group_cols in groups_to_run:
        print(f"{'─'*60}")
        print(f"Group: {group_name}  ({len(group_cols)} method(s))")
        print(f"{'─'*60}")

        fn = score_fn_map[group_name]

        for idx, (row, result) in enumerate(zip(all_rows, results), 1):
            kg1 = parse_triples(row.get('kg_1', ''))
            kg2 = parse_triples(row.get('kg_2', ''))

            if idx % 50 == 0 or idx == total:
                print(f"  [{idx:>4}/{total}] id={row.get('id', idx-1)}", flush=True)

            if not kg1 or not kg2:
                for col in group_cols:
                    result[col] = None
                continue

            try:
                scores = fn(kg1, kg2)
                result.update(scores)
            except Exception as e:
                print(f"  ERROR row {idx} (id={row.get('id')}): {e}")
                for col in group_cols:
                    result[col] = None

            gc.collect()

        # Free model allocations before next group
        gc.collect()
        print(f"  Done. Running gc.collect().")

        # Write checkpoint with all columns processed so far
        done_cols = all_done_cols + group_cols
        # Deduplicate while preserving order
        seen: set[str] = set()
        done_cols_deduped: list[str] = []
        for c in done_cols:
            if c not in seen:
                done_cols_deduped.append(c)
                seen.add(c)
        all_done_cols = done_cols_deduped

        checkpoint_fields = INPUT_FIELDNAMES + all_done_cols
        _write_csv(out_path, results, checkpoint_fields)
        print(f"  Checkpoint saved → {output_file}  ({len(all_done_cols)} method cols)\n")

    # ── Final write: all method columns, preserving order from ALL_METHODS ──
    final_cols = [m for m in ALL_METHODS if m in all_done_cols]
    _write_csv(out_path, results, INPUT_FIELDNAMES + final_cols)
    print(f"\n{'='*60}")
    print(f"Complete. Output: {output_file}")
    print(f"Method columns written: {len(final_cols)}")

    # Coverage summary
    print(f"\nCoverage ({total} rows):")
    for col in final_cols:
        valid = sum(1 for r in results if r.get(col) not in (None, ''))
        print(f"  {col:<40}  valid={valid}  errors={total-valid}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(
        script_dir,
        'Benchmarking_Pipeline', 'Results_All_Methods',
        'wikipedia_entity_swap_400_KGs_results.csv',
    )
    default_in = os.path.join(
        script_dir,
        'Benchmarking_Pipeline', 'Data',
        'wikipedia_entity_swap_400_KGs.csv',
    )

    parser = argparse.ArgumentParser(
        description='Sequential (group-by-group) KG similarity evaluation — low CPU footprint.'
    )
    parser.add_argument('--input',        default=default_in,
                        help='Input CSV file')
    parser.add_argument('--output',       default=default_out,
                        help='Output CSV file (also used as checkpoint)')
    parser.add_argument('--limit',        type=int, default=None,
                        help='Process only the first N rows (for testing)')
    parser.add_argument('--skip-groups',  nargs='+', default=None,
                        choices=GROUP_NAMES, metavar='GROUP',
                        help='Groups to skip (e.g. already computed in prior run)')
    parser.add_argument('--only-groups',  nargs='+', default=None,
                        choices=GROUP_NAMES, metavar='GROUP',
                        help='Run only these groups (overrides --skip-groups)')
    parser.add_argument('--list-groups',  action='store_true',
                        help='Print available group names and exit')
    args = parser.parse_args()

    if args.list_groups:
        print("Available method groups:")
        for name, cols in METHOD_GROUPS:
            print(f"  {name:<16} → {', '.join(cols)}")
        return

    process_sequential(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        skip_groups=args.skip_groups,
        only_groups=args.only_groups,
    )


if __name__ == '__main__':
    main()
