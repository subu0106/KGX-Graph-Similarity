import pandas as pd
import ast
import json

def parse_triplets(triplet_string):
    """
    Parse triplet string from dict format to simple list format.
    Input: "[{'source_node': {'name': 'X'}, 'relation': {'name': 'Y'}, 'target_node': {'name': 'Z'}}, ...]"
    Output: "[['X', 'Y', 'Z'], ...]"
    """
    try:
        # Try to evaluate the string as a Python literal
        triplets_dict = ast.literal_eval(triplet_string)

        # Convert to simple list format
        triplets_list = []
        for triplet in triplets_dict:
            source = triplet.get('source_node', {}).get('name', '')
            relation = triplet.get('relation', {}).get('name', '')
            target = triplet.get('target_node', {}).get('name', '')
            triplets_list.append([source, relation, target])

        return str(triplets_list)
    except Exception as e:
        print(f"Error parsing triplet: {e}")
        return "[]"

def transform_semantic_kg(input_file, output_file):
    """
    Transform semantic_kg_full.csv into the desired format.
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Create new dataframe with desired columns
    new_df = pd.DataFrame()

    print("Transforming graph 1...")
    new_df['graph_1'] = df['subgraph_triples'].apply(parse_triplets)

    print("Transforming graph 2...")
    new_df['graph_2'] = df['perturbed_subgraph_triples'].apply(parse_triplets)

    new_df['response1'] = df['response1']
    new_df['response2'] = df['response2']

    print("Extracting similarity scores...")
    new_df['similarity_score_ground'] = df['similarity']


    print(f"New shape: {new_df.shape}")
    print(f"\nSample of transformed data:")
    print(new_df.head(2))

    # Save to new CSV
    print(f"\nSaving to {output_file}...")
    new_df.to_csv(output_file, index=False)

    print("Transformation complete!")
    return new_df

if __name__ == "__main__":
    input_file = "/Users/subu/Desktop/FYP/KGX-Graph-Similarity/data/semantic_kg_full.csv"
    output_file = "/Users/subu/Desktop/FYP/KGX-Graph-Similarity/data/semantic_kg_transformed.csv"

    df = transform_semantic_kg(input_file, output_file)

    # Show a sample of the first row to verify format
    print("\n" + "="*80)
    print("Sample of first row:")
    print("="*80)
    print(f"Graph 1: {df['graph_1'].iloc[0][:200]}...")
    print(f"Graph 2: {df['graph_2'].iloc[0][:200]}...")
    print(f"Similarity: {df['similarity_score_ground'].iloc[0]}")
