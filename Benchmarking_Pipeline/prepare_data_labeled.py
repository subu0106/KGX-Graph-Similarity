#!/usr/bin/env python3
"""
Step 1 — Merge method scores with ground-truth labels.

For each dataset in Results_All_Methods/, find the matching labeled CSV
in src/evaluation/datasets/ and join on id == pair_id.

Output: Data_labeled/<dataset>_labeled.csv
Columns: id, label, perturbation_type, paragraph_1, paragraph_2,
         kg_1, kg_2, <all method score columns>

Usage:
    python prepare_data_labeled.py
"""

import pandas as pd
from pathlib import Path

HERE         = Path(__file__).parent
RESULTS_DIR  = HERE / 'Results_All_Methods'
LABELS_DIR   = Path(__file__).parent.parent.parent / 'evaluation' / 'datasets'
OUT_DIR      = HERE / 'Data_labeled'
OUT_DIR.mkdir(exist_ok=True)

# Mapping: results filename stem → label filename stem
# (strip _KGs_results from results name to get base, find matching label file)
DATASET_MAP = {
    'mrpc_400_KGs_results':                         'mrpc_400',
    'paws_wiki_400_KGs_results':                    'paws_wiki_400',
    'semantic_kg_codex_400_KGs_results':            'semantic_kg_codex_400',
    'semantic_kg_combined_400_KGs_results':         'semantic_kg_combined_400',
    'semantic_kg_findkg_400_KGs_results':           'semantic_kg_findkg_400',
    'semantic_kg_globi_400_KGs_results':            'semantic_kg_globi_400',
    'semantic_kg_oregano_400_KGs_results':          'semantic_kg_oregano_400',
    'sts12_400_KGs_results':                        'sts12_400',
    'wikipedia_entity_swap_400_KGs_results':        'wikipedia_entity_swap_400',
}

# pubmedqa has faithfulness_level instead of binary label — handled separately
PUBMEDQA_RESULTS = 'pubmedqa_ranked_faithfulness_400_KGs_results'
PUBMEDQA_LABELS  = 'pubmedqa_ranked_faithfulness_400'


def merge_dataset(results_stem, labels_stem):
    results_path = RESULTS_DIR / f'{results_stem}.csv'
    labels_path  = LABELS_DIR  / f'{labels_stem}.csv'

    if not results_path.exists():
        print(f'  SKIP — results not found: {results_path.name}')
        return None
    if not labels_path.exists():
        print(f'  SKIP — labels not found: {labels_path.name}')
        return None

    results = pd.read_csv(results_path)
    labels  = pd.read_csv(labels_path)

    results['id']    = results['id'].astype(int)
    labels['pair_id'] = labels['pair_id'].astype(int)

    merged = results.merge(
        labels[['pair_id', 'label', 'perturbation_type']],
        left_on='id', right_on='pair_id', how='inner'
    ).drop(columns=['pair_id'])

    out_path = OUT_DIR / f'{results_stem.replace("_results", "")}_labeled.csv'
    merged.to_csv(out_path, index=False)
    print(f'  {out_path.name}  ({len(merged)} rows, label dist: {dict(merged["label"].value_counts())})')
    return merged


def merge_pubmedqa():
    results_path = RESULTS_DIR / f'{PUBMEDQA_RESULTS}.csv'
    labels_path  = LABELS_DIR  / f'{PUBMEDQA_LABELS}.csv'

    if not results_path.exists() or not labels_path.exists():
        print(f'  SKIP pubmedqa — file missing')
        return None

    results = pd.read_csv(results_path)
    labels  = pd.read_csv(labels_path)

    results['id']     = results['id'].astype(int)
    labels['pair_id'] = labels['pair_id'].astype(int)

    # Convert faithfulness_level (L1=high, L2=medium, L3=low) → binary label
    # L1 (faithful) = 1, L2/L3 = 0
    def faith_to_binary(lvl):
        if isinstance(lvl, str):
            lvl = lvl.strip().upper()
        return 1 if lvl in ('L1', 'HIGH', '1') else 0

    labels['label'] = labels['faithfulness_level'].apply(faith_to_binary)
    labels['perturbation_type'] = labels.get('change_type', labels.get('perturbation_type', 'unknown'))

    merged = results.merge(
        labels[['pair_id', 'label', 'perturbation_type']],
        left_on='id', right_on='pair_id', how='inner'
    ).drop(columns=['pair_id'])

    out_path = OUT_DIR / f'{PUBMEDQA_RESULTS.replace("_results", "")}_labeled.csv'
    merged.to_csv(out_path, index=False)
    print(f'  {out_path.name}  ({len(merged)} rows, label dist: {dict(merged["label"].value_counts())})')
    return merged


if __name__ == '__main__':
    print('Preparing labeled datasets...\n')
    for results_stem, labels_stem in DATASET_MAP.items():
        print(f'Processing: {results_stem}')
        merge_dataset(results_stem, labels_stem)

    print(f'\nProcessing: {PUBMEDQA_RESULTS}')
    merge_pubmedqa()

    print(f'\nDone. Labeled files saved to: {OUT_DIR}')
