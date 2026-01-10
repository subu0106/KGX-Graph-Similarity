import networkx as nx
import numpy as np
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, NeighborhoodSubgraphPairwiseDistance
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

sbert_model = SentenceTransformer('paraphrase-MPNet-base-v2')

def get_sbert_embedding(label):
    embedding = sbert_model.encode(label, convert_to_tensor=True)
    return embedding.detach().cpu().numpy()

def choose_representative(cluster):
    return min(cluster, key=len)

def is_number(label):
    try:
        float(label)
        return True
    except ValueError:
        return False

def cluster_data(nodes):
    new_nodes = {}
    
    numeric_labels = {label: label for label in nodes if is_number(label)}
    non_numeric_labels = [label for label in nodes if not is_number(label)]

    embeddings = np.array([get_sbert_embedding(label) for label in non_numeric_labels])
    cluster = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.35)
    labels = cluster.fit_predict(embeddings)

    clusters = {}
    for label, cluster_id in zip(non_numeric_labels, labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(label)

    for cluster_id, cluster in clusters.items():
        representative = choose_representative(cluster)
        for label in cluster:
            new_nodes[label] = representative

    new_nodes.update(numeric_labels)
    
    return new_nodes

def create_networkx_graph(triple_list):
    G = nx.Graph()
    for triple in triple_list:
        subject, predicate, obj = triple
        G.add_edge(subject.lower(), obj.lower(), relation=predicate.lower())
        G.nodes[subject.lower()]['label'] = subject.lower()
        G.nodes[obj.lower()]['label'] = obj.lower()
    return G

# def relabel_graph(nx_graph, label_clusters):
#     for node, data in nx_graph.nodes(data=True):
#         original_label = data['label']
#         nx_graph.nodes[node]['label'] = label_clusters[original_label]

#     for u, v, data in nx_graph.edges(data=True):
#         original_label = data['relation']
#         nx_graph.edges[u, v]['relation'] = label_clusters[original_label]

def relabel_graph(nx_graph, label_clusters):
    new_graph = nx.Graph()

    node_to_cluster = {}

    for node, data in nx_graph.nodes(data=True):
        original_label = data['label']
        curr_cluster = label_clusters[original_label]
        node_to_cluster[node] = curr_cluster
        new_graph.add_node(curr_cluster)

    for u, v, data in nx_graph.edges(data=True):
        new_u = node_to_cluster[u]
        new_v = node_to_cluster[v]
        
        if new_graph.has_edge(new_u, new_v):
            pass
        else:
            new_graph.add_edge(new_u, new_v, relation=data['relation'])

    return new_graph

def convert_to_grakel_graph(nx_graph):
    node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
    
    edge_labels = {(u, v): data.get('relation', 'default_relation') for u, v, data in nx_graph.edges(data=True)}

    edges = {(u, v): 1 for u, v in nx_graph.edges()}

    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)

def determine_number_of_clusters(labels):
    n_labels = len(labels)
    return max(3, int(np.sqrt(n_labels)))

def normalize_label(label):
    return label.lower()

def cluster_labels(labels):
    normalized_labels = [normalize_label(label) for label in labels]
    embeddings = np.vstack([get_sbert_embedding(label) for label in normalized_labels])
    n_clusters = determine_number_of_clusters(labels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clustered_labels = {label: f'cluster_{kmeans.labels_[i]}' for i, label in enumerate(labels)}
    return clustered_labels

def get_triple_embedding(triple):
    triple_text = ' '.join(map(str, triple))
    embedding = get_sbert_embedding(triple_text)
    return embedding

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def match_and_filter_triples(kg1_triples, kg2_triples):
    picked_triples = []
    for triple in kg1_triples:
        curr_kg1_embedding = get_triple_embedding(triple)
        similarities = []
        for other_triple in kg2_triples:
            curr_kg2_embedding = get_triple_embedding(other_triple)
            curr_similarity = cosine_similarity(curr_kg1_embedding, curr_kg2_embedding)
            similarities.append((curr_similarity, other_triple))

        chosen_triple = max(similarities, key=lambda x: x[0])
        if chosen_triple[1] not in picked_triples:
            picked_triples.append(chosen_triple[1])
    return picked_triples

def calculate_similarity(kg1_triples, kg2_triples):

    kg1_triples = [sublist for sublist in kg1_triples if len(sublist) == 3]
    kg2_triples = [sublist for sublist in kg2_triples if len(sublist) == 3]


    if len(kg2_triples) == 0:
        return 0
    if len(kg1_triples) == 0:
        return 0

    filtered_kg2_triples = match_and_filter_triples(kg1_triples, kg2_triples)

    kg1_graph = create_networkx_graph(kg1_triples)
    kg2_graph = create_networkx_graph(filtered_kg2_triples)

    all_labels = set(nx.get_node_attributes(kg1_graph, 'label').values()) | \
                set(nx.get_node_attributes(kg2_graph, 'label').values()) | \
                set(nx.get_edge_attributes(kg1_graph, 'relation').values()) | \
                set(nx.get_edge_attributes(kg2_graph, 'relation').values())

    label_clusters = cluster_data(all_labels)
    #print(f"Label clusters: {label_clusters}")


    relabelled_kg1 = relabel_graph(kg1_graph, label_clusters)
    relabelled_kg2 = relabel_graph(kg2_graph, label_clusters)

    kg1_grakel = convert_to_grakel_graph(relabelled_kg1)
    kg2_grakel = convert_to_grakel_graph(relabelled_kg2)

    nspd_kernel = WeisfeilerLehman(n_jobs=2, normalize=True)
    kernel_matrix = nspd_kernel.fit_transform([kg1_grakel, kg2_grakel])

    similarity = kernel_matrix[0, 1]
    return similarity, relabelled_kg1.edges(data=True), relabelled_kg2.edges(data=True)


def extract_all_labels(triples):
    """
    Extract all unique labels from triples (subjects, predicates, objects)

    Args:
        triples: List of triples [[s, p, o], ...]

    Returns:
        List of unique lowercase labels
    """
    labels = set()
    for triple in triples:
        if len(triple) == 3:
            labels.add(triple[0].lower())  # subject
            labels.add(triple[1].lower())  # predicate
            labels.add(triple[2].lower())  # object
    return list(labels)


def calculate_gaussian_feature_similarity(kg1_triples, kg2_triples, sigma=1.0):
    """
    Compute Gaussian kernel similarity on SBERT embeddings

    Formula: K(x1, x2) = exp(-||emb1 - emb2||^2 / (2 * sigma^2))

    Args:
        kg1_triples: First set of triples
        kg2_triples: Second set of triples
        sigma: Kernel width parameter (default: 1.0)

    Returns:
        float: Similarity score (0-1)
    """
    # Filter valid triples
    kg1_triples = [sublist for sublist in kg1_triples if len(sublist) == 3]
    kg2_triples = [sublist for sublist in kg2_triples if len(sublist) == 3]

    if len(kg1_triples) == 0 or len(kg2_triples) == 0:
        return 0.0

    # Extract all labels
    labels1 = extract_all_labels(kg1_triples)
    labels2 = extract_all_labels(kg2_triples)

    if not labels1 or not labels2:
        return 0.0

    # Get SBERT embeddings
    embeddings1 = np.array([get_sbert_embedding(label) for label in labels1])
    embeddings2 = np.array([get_sbert_embedding(label) for label in labels2])

    # Compute pairwise Gaussian kernel
    # For each label in graph1, find best match in graph2
    similarities = []
    for emb1 in embeddings1:
        max_sim = 0
        for emb2 in embeddings2:
            # Euclidean distance
            dist = np.linalg.norm(emb1 - emb2)
            # Gaussian kernel
            sim = np.exp(-dist**2 / (2 * sigma**2))
            max_sim = max(max_sim, sim)
        similarities.append(max_sim)

    # Average of best matches
    return float(np.mean(similarities))


def calculate_composite_similarity(kg1_triples, kg2_triples, alpha=0.6, sigma=1.0):
    """
    Composite kernel: combines structural and semantic similarity

    Formula: composite = alpha * structural + (1-alpha) * semantic

    Args:
        kg1_triples: First set of triples
        kg2_triples: Second set of triples
        alpha: Weight for structural similarity (default: 0.6)
               1.0 = pure structural (original KEA)
               0.0 = pure semantic
        sigma: Gaussian kernel width (default: 1.0)

    Returns:
        dict: {
            'composite': combined score,
            'structural': WL kernel score,
            'semantic': Gaussian kernel score
        }
    """
    # Structural similarity (existing KEA with WL kernel)
    structural_sim, _, _ = calculate_similarity(kg1_triples, kg2_triples)

    # Semantic similarity (new Gaussian kernel)
    semantic_sim = calculate_gaussian_feature_similarity(kg1_triples, kg2_triples, sigma)

    # Composite
    composite_sim = alpha * structural_sim + (1 - alpha) * semantic_sim

    return {
        'composite': float(composite_sim),
        'structural': float(structural_sim),
        'semantic': float(semantic_sim)
    }


# kg1_triples = [['Russian fighter jet', 'intercepted', 'U.S. reconnaissance plane'], ['Pentagon', 'says', 'incident occurred in international airspace north of Poland'], ['Russian jet', 'flew within', '100 feet of RC-135U'], ['RC-135U', 'was intercepted by', 'Russian SU-27 Flanker'], ['United States', 'is complaining about', 'incident']]

# kg2_triples = [['Russian fighter jet', 'intercepted', 'U.S. reconnaissance plane'], ['United States', 'is complaining to', 'Moscow about the incident'], ['U.S. RC-135U', 'was flying over', 'Baltic Sea'], ['U.S. RC-135U', 'was intercepted by', 'Russian SU-27 Flanker'], ['Pentagon', 'said', 'incident occurred in international airspace north of Poland'], ['U.S. crew', 'believed', "Russian pilot's actions were unsafe and unprofessional"], ['Russian jet', 'flew around', 'U.S. plane several times'], ['Pentagon', 'will file', 'appropriate petition through diplomatic channels with Russia'], ['U.S. has complained about', 'incident involving', 'RC-135U and SU-27'], ['Russian jet', 'flew within', '100 feet of RC-135U over Sea of Okhotsk']]
# print(calculate_similarity(kg1_triples, kg2_triples))

# kg2_triples = [['Donald Sterling', 'nationality', 'American'], ['Donald Sterling', 'occupation', 'Businessman'], ['Donald Sterling', 'spouse', 'Rochelle Sterling'], ['Donald Sterling', 'former companion', 'V. Stiviano'], ['Donald Sterling', 'owned', 'Los Angeles Clippers'], ['V. Stiviano', 'received gifts from', 'Donald Sterling'], ['V. Stiviano', 'ordered to pay back', '$2.6 million'], ['Rochelle Sterling', 'sued', 'V. Stiviano'], ['Rochelle Sterling', 'accused', 'V. Stiviano of targeting wealthy older men'], ['Magic Johnson', 'associated with', 'V. Stiviano'], ['Magic Johnson', 'mentioned in', "Donald Sterling's racist remarks"], ['Adam Silver', 'banned', 'Donald Sterling from the NBA'], ['Adam Silver', 'fined', 'Donald Sterling $2.5 million'], ['Los Angeles Clippers', 'formerly owned by', 'Donald Sterling'], ['Los Angeles Clippers', 'games', 'attended by Magic Johnson'], ['Ferrari', 'owned by', 'V. Stiviano'], ['Bentleys', 'owned by', 'V. Stiviano'], ['Range Rover', 'owned by', 'V. Stiviano'], ['$1.8 million duplex', 'owned by', 'V. Stiviano'], ['$391 Easter bunny costume', 'owned by', 'V. Stiviano'], ['$299 two-speed blender', 'owned by', 'V. Stiviano'], ['$12 lace thong', 'owned by', 'V. Stiviano'], ['Donald Sterling', 'made fortune in', 'real estate'], ['Donald Sterling', 'recorded making racist remarks', 'audio recording'], ['Donald Sterling', 'downfall', 'after audio recording surfaced'], ['V. Stiviano', 'countered', 'that she never took advantage of Donald Sterling'], ['Shelly Sterling', 'thrilled with court decision', 'Tuesday'], ["Pierce O'Donnell", 'represented', 'Shelly Sterling'], ['KABC', 'reported on', 'court decision'], ['TMZ', 
# 'first posted', "audio recording of Donald Sterling's racist remarks"], ['Los Angeles Times', 'reported on', "V. Stiviano's gifts from Donald Sterling"], ['Dottie Evans', 'contributed to', 'CNN report'], ['CNN', 'reported on', "Donald Sterling's racist remarks and court decision"]]
# kg1_triples = [['Donald Sterling', 'nationality', 'American'], ['Donald Sterling', 'occupation', 'Businessman'], ['Donald Sterling', 'spouse', 'Rochelle Sterling'], ['Donald Sterling', 'former companion', 'V. Stiviano'], ['Donald Sterling', 'owned', 'Los Angeles Clippers'], ['V. Stiviano', 'received gifts from', 'Donald Sterling'], ['V. Stiviano', 'ordered to pay back', '$2.6 million'], ['Rochelle Sterling', 'sued', 'V. Stiviano'], ['Rochelle Sterling', 'accused', 'V. Stiviano of targeting wealthy older men'], ['Magic Johnson', 'associated with', 'V. Stiviano'], ['Magic Johnson', 'mentioned in', "Donald Sterling's racist remarks"], ['Adam Silver', 'banned', 'Donald Sterling from the NBA'], ['Adam Silver', 'fined', 'Donald Sterling $2.5 million'], ['Los Angeles Clippers', 'formerly owned by', 'Donald Sterling'], ['Los Angeles Clippers', 'games', 'attended by Magic Johnson'], ['Ferrari', 'owned by', 'V. Stiviano'], ['Bentleys', 'owned by', 'V. Stiviano'], ['Range Rover', 'owned by', 'V. Stiviano'], ['$1.8 million duplex', 'owned by', 'V. Stiviano'], ['$391 Easter bunny costume', 'owned by', 'V. Stiviano'], ['$299 two-speed blender', 'owned by', 'V. Stiviano'], ['$12 lace thong', 'owned by', 'V. Stiviano'], ['Donald Sterling', 'made fortune in', 'real estate'], ['Donald Sterling', 'recorded making racist remarks', 'audio recording'], ['Donald Sterling', 'downfall', 'after audio recording surfaced'], ['V. Stiviano', 'countered', 'that she never took advantage of Donald Sterling'], ['Shelly Sterling', 'thrilled with court decision', 'Tuesday'], ["Pierce O'Donnell", 'represented', 'Shelly Sterling'], ['KABC', 'reported on', 'court decision'], ['TMZ', 
# 'first posted', "audio recording of Donald Sterling's racist remarks"], ['Los Angeles Times', 'reported on', "V. Stiviano's gifts from Donald Sterling"], ['Dottie Evans', 'contributed to', 'CNN report'], ['CNN', 'reported on', "Donald Sterling's racist remarks and court decision"]]
# print(calculate_similarity(kg1_triples, kg2_triples))

# kg1_triples = [['Alan Dinehart', 'born', '1888'], ['Alan Dinehart', 'died', '1944'], ['Alan Dinehart', 'nationality', 'American'], ['Alan Dinehart', 'occupation', 'actor']]
# kg2_triples = [['Alan Dinehart', 'nationality', 'American'], ['Alan Dinehart', 'birth year', '1889'], ['Alan Dinehart', 'death year', '1944'], ['Alan Dinehart', 'occupation', 'actor'], ['Alan Dinehart', 'occupation', 'actor'], ['April', 'is', 'fourth month in the Julian and Gregorian calendars'], ['October', 'instance of', 'month'], ['October', 'in', 'Julian calendar'], ['October', 'in', 'Gregorian calendar'], ['actor', 'is', 'person'], ['actor', 'acts in', 'dramatic or comic production'], ['actor', 'works in', 'film'], ['actor', 'works in', 'television'], ['actor', 'works in', 'theatre'], ['actor', 'works in', 'radio'], ['film sequence', 'instance of', 'sequence of images'], ['film sequence', 'give the impression of', 'movement'], ['film sequence', 'stored on', 'film stock'], ['film sequence', 'instance of', 'sequence of images'], ['film sequence', 'give the impression of', 'movement'], ['film sequence', 'stored on', 'film stock'], ['Leading man', 'instance of', 'male lead'], ['Leading man', 'in', 'film or play'], ['Suave', 'represents', 'more than 100 products including shampoo, lotions, soaps and deodorant'], ['New York City', 'most populous city in', 'United States'], ['career', 'instance of', "individual's journey"], ['career', 'related to', 'learning'], ['career', 'related to', 'work'], ['career', 'related to', 'other aspects of life'], ['stagecoach', 'type of', 'covered wagon'], ['film', 'instance of', 'sequence of images that give the impression of movement'], ['film', 'stored on', 'film stock'], ['Leading man', 'instance of', 'male lead'], ['Leading man', 'in', 'film or play'], ['silent film', 'instance of', 'film'], ['silent film', 'characterized by', 'no synchronized recorded dialogue'], ['silent film', 'subclass of', 'film'], ['silent film', 'subclass of', 'film'], ['silent film', 'subclass of', 'film'], ['silent film', 'opposite of', 'sound film'], ['silent film', 'subclass of', 'film'], ['silent film', 'subclass of', 'film'], ['lead chemical element', 'has symbol', 'Pb'], ['lead chemical element', 'has atomic number', '82'], ['star', 'instance of', 'astronomical object'], ['star', 'consists of', 'luminous spheroid of plasma'], ['star', 'held together by', 'its own gravity'], ['Mary Pickford', 'nationality', 'Canada'], ['Mary Pickford', 'occupation', 'actress'], ['Mary Pickford', 'occupation', 'producer'], ['Mary Pickford', 'born', '1892'], ['Mary Pickford', 'died', '1979'], ['Mary Pickford', 'occupation', 'actor'], ['Lillian Gish', 'occupation', 'actor'], ['Lillian Gish', 'place of death', 'New York City'], ['Clara Bow', 'nationality', 'American'], ['Clara Bow', 'birth year', '1905'], ['Clara Bow', 'death year', '1965'], ['Clara Bow', 'occupation', 'actress'], ['Clara Bow', 'occupation', 'actor'], ['number', 'instance of', 'mathematical object'], ['number', 'used to', 'count'], ['number', 'used to', 'label'], ['number', 'used to', 'measure'], ['television', 'instance of', 'western'], ['television', 'instance of', 'television genre'], ['The Covered Wagon', 'instance of', '1923 film'], ['The Covered Wagon', 'directed by', 'James Cruze'], ['The Covered Wagon', 'instance of', 'film'], ['The Covered Wagon', 'instance of', 'film'], ['The Covered Wagon', 'instance of', 'film'], ['The Covered Wagon', 'genre', 'silent film'], ['The Covered Wagon', 'instance of', 'film'], ['The Covered Wagon', 'instance of', 'film'], ['Lou Gehrig', 'instance of', 'American baseball player'], ['1930s decade', 'part of', '20th century'], ['1930s decade', 'feature', 'Great Depression'], ['1930s decade', 'feature', 'World War II'], ['The Invisible Man', 'created by', 'James Whale'], ['The Invisible Man', 'type of', 'film'], ['The Invisible Man', 'instance of', 'film'], ['The Invisible Man', 'instance of', 'film'], ['The Invisible Man', 'instance of', 'film'], ['The Invisible Man', 'instance of', 'film'], ['The Invisible Man', 'instance of', 'film'], ['The Little Minister', 'created by', 'Richard Wallace'], ['The Little Minister', 'type of', 'film'], ['The Little Minister', 'released in', '1934'], ['The Little Minister', 'instance of', 'film'], ['The Little Minister', 'instance of', 'film'], ['The Little Minister', 'instance of', 'film'], ['The Little Minister', 'instance of', 'film'], ['The Little Minister', 'instance of', 'film'], ['sound film', 'instance of', 'motion picture'], ['sound film', 'characterized by', 'synchronized sound'], ['sound film', 'subclass of', 'film'], ['sound film', 'subclass of', 'film'], ['sound film', 'subclass of', 'film'], ['sound film', 'opposite of', 'silent film'], ['sound film', 'subclass of', 'film'], ['sound film', 'subclass of', 'film'], ['number', 'instance of', 'mathematical object'], ['number', 'used to', 'count'], ['number', 'used to', 'label'], ['number', 'used to', 'measure'], ['film sequence', 'instance of', 'sequence of images'], ['film sequence', 'give the impression of', 'movement'], ['film sequence', 'stored on', 'film stock'], ['The Big Broadcast 1932 film', 'created by', 'Frank Tuttle'], ['The Big Broadcast', 'instance of', 'film'], ['The Big Broadcast', 'instance of', 'film'], ['The Big Broadcast', 'instance of', 'film'], ['The Big Broadcast', 'instance of', 'film'], ['The Big Broadcast', 'instance of', 'film'], ['film sequence', 'instance of', 'sequence of images'], ['film sequence', 'give the impression of', 'movement'], ['film sequence', 'stored on', 'film stock'], ['death', 'instance of', 'permanent cessation of vital functions']]
# print(calculate_similarity(kg1_triples, kg2_triples))

# claim = [
#     ["Marie Curie", "discovered", "Radium"],
#     ["Marie Curie", "won", "Nobel Prize in Physics"],
#     ["Marie Curie", "won", "Nobel Prize in Chemistry"]
# ]

# Marie Curie found Radium, and was awarded the Nobel Prize in Chemistry and Physics
# evidence = [
#     ["Marie Curie", "found", "Radium"],
#     ["Marie Curie", "received", "Nobel Prize in Physics"],
#     ["Marie Curie", "was awarded", "Nobel Prize in Chemistry"]
# ]

# Albert Einstein found Radium. Marie Curie received the Grammy and was also awarded the Emmy.
# evidence = [
#     ["Albert Einstein", "found", "Radium"],
#     ["Marie Curie", "received", "Grammy"],
#     ["Marie Curie", "was awarded", "Emmy"]
# ]


# similarity_score = calculate_similarity(claim, evidence)
# print(f"Graph similarity score: {similarity_score}")





# claim = [['Malcolm Brogdon', 'born', 'December 11, 1992'], ['Malcolm Brogdon', 'nationality', 'American'], ['Malcolm Brogdon', 'occupation', 'basketball player'], ['Malcolm Brogdon', 'plays for', 'Indiana Pacers'], ['Malcolm Brogdon', 'league', 'National Basketball Association'], ['Malcolm Brogdon', 'played for', 'Virginia Cavaliers'], ['Malcolm Brogdon', 'awards', 'ACC Player of the Year'], ['Malcolm Brogdon', 'awards', 'All-American'], ['Malcolm Brogdon', 'year', '2016'], ['Malcolm Brogdon', 'drafted by', 'Milwaukee Bucks'], ['Malcolm Brogdon', 'pick', '36th overall'], ['Malcolm Brogdon', 'awards', 'Rookie of the Year'], ['Malcolm Brogdon', 'year', '2017'], ['Malcolm Brogdon', 'traded to', 'Indiana Pacers'], ['Malcolm Brogdon', 'NBA All-Star', 'two-time'], ['Malcolm Brogdon', 'named to', 'All-Defensive Second Team'], ['Malcolm Brogdon', 'year', '2019'], ['Malcolm Brogdon', 'skills', 'defensive prowess'], ['Malcolm Brogdon', 'skills', 'long range shooting'], ['Malcolm Brogdon', 'advocacy', 'social justice'], ['Malcolm Brogdon', 'involved in', 'initiatives for racial equality']]
# evidence = [['Malcolm Moses Adams Brogdon', 'born on', 'December 11, 1992'], ['Malcolm Brogdon', 'is', 'American basketball player'], ['Malcolm Brogdon', 'member of sports team', 'Indiana Pacers'], ['Malcolm Brogdon', 'league', 'National Basketball Association'], ['Malcolm Moses Adams Brogdon', 'played college basketball for', 'the Virginia Cavaliers under Tony Bennett'], ['Malcolm Moses Adams Brogdon', 'was named', 'the Atlantic Coast Conference (ACC) Player of the Year'], ['Malcolm Brogdon', 'family name', 'Brogdon'], ['Malcolm Brogdon', 'drafted by', 'Milwaukee Bucks'], ['Malcolm Moses Adams Brogdon', 'was selected by', 'the Milwaukee Bucks with the 36th overall pick in the 2016 NBA draft'], ['Malcolm Moses Adams Brogdon', 'won', 'the NBA Rookie of the Year Award'], ['Malcolm Moses Adams Brogdon', 'was traded to', 'the Indiana Pacers'], ['Malcolm Moses Adams Brogdon', 'was named', 'a consensus second-team All-American in 2014–15'], ['Malcolm Brogdon', 'sport', 'basketball'], ['shooter', 'must have a rifle with good precision to succeed at long range shooting'], ['Malcolm Moses Adams Brogdon', 'has', 'a Masters Degree in Public Policy from the Batten School of Leadership and Public Policy at the University of Virginia'], ['Malcolm Moses Adams Brogdon', 'founded', 'his own nonprofit, The Brogdon Family Foundation, in 2021']]
# claim = [sublist for sublist in claim if len(sublist) == 3]
# evidence = [sublist for sublist in evidence if len(sublist) == 3]
# similarity_score = calculate_similarity(claim, evidence)
# print(f"Graph similarity score: {similarity_score}")


# claim = [["Apples", "type of", "fruit"], ["Apples", "grow on", "trees"]]
# evidence = [["Apples", "type of", "fruit"], ["Apples", "grow in", "tree"]]
# print(calculate_similarity(claim, evidence))