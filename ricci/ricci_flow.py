import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import networkx as nx
import itertools
from collections import defaultdict


# Compute degrees of simplices
def compute_degrees(sc):
    degrees = {}
    max_dim = max(sc.simplices.keys())
    for dim in sc.simplices:
        if dim < max_dim:
            for simplex in sc.simplices[dim]:
                cofaces = get_cofaces(sc, simplex)
                degrees[simplex] = len(cofaces)
    return degrees


# Get cofaces of a simplex
def get_cofaces(sc, simplex):
    dim = len(simplex) - 1
    cofaces = []
    if dim + 1 in sc.simplices:
        for coface in sc.simplices[dim + 1]:
            if set(simplex).issubset(coface):
                cofaces.append(coface)
    return cofaces


# Compute probability measures for simplices
def compute_probability_measures(sc, degrees):
    m = {}
    for dim in sc.simplices:
        if dim < max(sc.simplices.keys()):
            for simplex in sc.simplices[dim]:
                deg = degrees[simplex]
                m_F = defaultdict(float)
                cofaces = get_cofaces(sc, simplex)
                for coface in cofaces:
                    weight = sc.weights[coface]
                    faces = [tuple(sorted(face)) for face in itertools.combinations(coface, dim + 1) if face != simplex]
                    for face in faces:
                        m_F[face] += weight / ((dim + 1) * deg)
                m[simplex] = m_F
    return m

# Compute the curvative
def compute_ricci_curvature(sc, m):
    curvature = {}
    for dim in sc.simplices:
        if dim < max(sc.simplices.keys()):
            for simplex in sc.simplices[dim]:
                adjacent_simplices = get_adjacent_simplices(sc, simplex)
                kappa = {}
                for adj in adjacent_simplices:
                    W = wasserstein_distance(m[simplex], m[adj])
                    kappa[adj] = 1 - W
                curvature[simplex] = kappa
    return curvature

# Compute the Wasserstein distance
def wasserstein_distance(m_F, m_G):
    keys = set(m_F.keys()).union(set(m_G.keys()))
    distance = 0.0
    for k in keys:
        distance += abs(m_F.get(k, 0) - m_G.get(k, 0))
    distance *= 0.5  # Since the ground distance is 1
    return distance


def get_adjacent_simplices(sc, simplex):
    adjacent = set()
    cofaces = get_cofaces(sc, simplex)
    for coface in cofaces:
        faces = [tuple(sorted(face)) for face in itertools.combinations(coface, len(simplex)) if face != simplex]
        for face in faces:
            adjacent.add(face)
    return list(adjacent)

# Main function for Ricci flow community detection
def ricci_flow_community_detection(sc,
                                    T=10,
                                    delta=0.01,
                                    ground_truth=None, 
                                    min_theta_range=0.3, 
                                    theta_step=0.001
                                    ):
    degrees = compute_degrees(sc)
    nmi_scores = []
    modularity_scores = []
    ari_scores = []

    # Run Ricci flow iterations
    for t in range(T):
        print(f"Iteration {t + 1}/{T}")
        m = compute_probability_measures(sc, degrees)
        # Update weights based on the Ricci curvature computation

        # Compute Ricci curvature
        curvature = compute_ricci_curvature(sc, m)

        update_weights(sc, curvature, delta)

    # Compute weight range and theta values
    max_dim = max(sc.simplices.keys())
    all_weights = [w for s, w in sc.weights.items() if len(s) == max_dim]
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    print(f"Weight range after Ricci flow: min={min_weight}, max={max_weight}")

    # Define theta values based on weight range
    theta_values = np.arange(min_weight - min_theta_range, max_weight, theta_step)

    # Store original simplices and weights
    original_simplices = {dim: sc.simplices[dim].copy() for dim in sc.simplices}
    original_weights = sc.weights.copy()

    # For each theta, perform network surgery and compute metrics
    for theta in theta_values:
        print(f"\nApplying weight cutoff theta = {theta}")
        # Reset simplices and weights
        sc.simplices = {dim: original_simplices[dim].copy() for dim in original_simplices}
        sc.weights = original_weights.copy()

        # Network surgery (removal of simplices based on thresholding)
        simplices_to_remove = [simplex for simplex, weight in sc.weights.items() if weight > theta and len(simplex) > 1]
        for simplex in simplices_to_remove:
            dim = len(simplex) - 1
            if simplex in sc.simplices[dim]:
                sc.simplices[dim].remove(simplex)
            del sc.weights[simplex]

        # Identify communities
        communities = identify_communities(sc)

        print(f"Detected {len(communities)} communities at theta={theta}")

        # If ground truth is provided, compute NMI and ARI
        if ground_truth is not None:
            detected_labels = node_labels_from_communities_dict(communities)
            ground_truth_node_ids = set(ground_truth.keys())
            detected_node_ids = set(detected_labels.keys())
            common_node_ids = ground_truth_node_ids & detected_node_ids
            aligned_node_ids = sorted(common_node_ids)

            aligned_ground_truth_labels = [ground_truth[node_id] for node_id in aligned_node_ids]
            aligned_detected_labels = [detected_labels[node_id] for node_id in aligned_node_ids]

            valid_indices = [i for i, label in enumerate(aligned_ground_truth_labels) if label != -1]

            filtered_ground_truth_labels = [aligned_ground_truth_labels[i] for i in valid_indices]
            filtered_detected_labels = [aligned_detected_labels[i] for i in valid_indices]
            

            # Compute NMI and ARI
            nmi = normalized_mutual_info_score(filtered_ground_truth_labels, filtered_detected_labels)
            nmi_scores.append(nmi)
            ari = adjusted_rand_score(filtered_ground_truth_labels, filtered_detected_labels)
            ari_scores.append(ari)
            print(f"NMI Score: {nmi}")
            print(f"ARI Score: {ari}")
        else:
            nmi_scores.append(None)
            ari_scores.append(None)

        try:
          modularity = compute_modularity(sc, communities)
        except:
          modularity = 0        
          
        modularity_scores.append(modularity)  
        print(f"Modularity: {modularity}")

    return nmi_scores, modularity_scores, theta_values, ari_scores


# Update weights based on Ricci curvature
def update_weights(sc, curvature, delta):
    new_weights = sc.weights.copy()
    for dim in sc.simplices:
        if dim < max(sc.simplices.keys()) - 1:
            for simplex in sc.simplices[dim]:
                kappa_values = list(curvature.get(simplex, {}).values())
                if kappa_values:
                    avg_kappa = np.mean(kappa_values)
                    cofaces = get_cofaces(sc, simplex)
                    for coface in cofaces:
                        new_weights[coface] *= np.exp(-delta * avg_kappa / (len(simplex)))
    sc.weights = new_weights

# Identify communities from simplicial complex
def identify_communities(sc):
    G = nx.Graph()
    G.add_nodes_from([v[0] for v in sc.simplices[0]])
    G.add_edges_from(sc.simplices[1])

    components = list(nx.connected_components(G))
    return components


# Compute modularity of a community partition
def compute_modularity(sc, communities):
    G = nx.Graph()
    G.add_nodes_from([v[0] for v in sc.simplices[0]])
    G.add_edges_from(sc.simplices[1])

    community_list = []
    for community in communities:
        community_list.append(set(community))

    modularity = nx.algorithms.community.modularity(G, community_list)
    return modularity


# Convert communities to node labels
def node_labels_from_communities_dict(communities):
    node_labels = {}
    for label, community in enumerate(communities):
        for node in community:
            node_labels[node] = label
    return node_labels


# Convert detected communities to node labels
def node_labels_from_communities(communities, num_nodes=None):
    node_labels = {}
    for label, community in enumerate(communities):
        for node in community:
            node_labels[node] = label

    if num_nodes is not None:
        labels = [node_labels.get(node, -1) for node in range(num_nodes)]  # Default label is -1 for unlabeled nodes
    else:
        labels = [node_labels[node] for node in sorted(node_labels.keys())]

    return labels