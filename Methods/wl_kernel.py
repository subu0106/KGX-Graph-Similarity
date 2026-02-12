"""
Pure Weisfeiler-Lehman Kernel
Reference: Shervashidze et al. "Weisfeiler-Lehman Graph Kernels" (JMLR 2011)
"""

import networkx as nx
from grakel import Graph
from grakel.kernels import WeisfeilerLehman


def calculate_pure_wl_kernel_similarity(triples1, triples2):
    """Calculate pure WL kernel similarity (structural only, no semantic clustering)"""

    def create_networkx_graph(triple_list):
        G = nx.Graph()
        for triple in triple_list:
            if len(triple) == 3:
                subject, predicate, obj = triple
                G.add_edge(subject.lower(), obj.lower(), relation=predicate.lower())
                G.nodes[subject.lower()]['label'] = subject.lower()
                G.nodes[obj.lower()]['label'] = obj.lower()
        return G

    def convert_to_grakel_graph(nx_graph):
        node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
        edge_labels = {(u, v): data.get('relation', 'default_relation') for u, v, data in nx_graph.edges(data=True)}
        edges = {(u, v): 1 for u, v in nx_graph.edges()}
        return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)

    try:
        kg1_graph = create_networkx_graph(triples1)
        kg2_graph = create_networkx_graph(triples2)

        kg1_grakel = convert_to_grakel_graph(kg1_graph)
        kg2_grakel = convert_to_grakel_graph(kg2_graph)

        wl_kernel = WeisfeilerLehman(n_jobs=2, normalize=True)
        kernel_matrix = wl_kernel.fit_transform([kg1_grakel, kg2_grakel])

        similarity = kernel_matrix[0, 1]
        return float(similarity)
    except Exception as e:
        print(f"Error in pure WL kernel: {e}")
        return None
