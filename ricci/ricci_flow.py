import numpy as np
import networkx as nx
import ot                      
import itertools
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt



# Precompute ground metric D[(F,G)]
def compute_ground_metric(sc, d):
    D = {}
    simplices = sc.simplices[d]
    for u in simplices:
        for v in simplices:
            if u == v:
                D[(u, v)] = 0.0
            else:
                intersection_size = len(set(u) & set(v))
                # The more intersection, the smaller the distance
                D[(u, v)] = 1.0 - intersection_size / (d + 1)
    return D

def build_all_ground_metrics(sc):
    max_dim = max(sc.simplices.keys())
    D_dict = {}
    # for each dimension i where we compute curvature (i < max_dim):
    for i in range(max_dim):
        D_dict[i] = compute_ground_metric(sc, i)
    return D_dict



def compute_degrees(sc):
    degrees = {}
    max_dim = max(sc.simplices.keys())
    for dim in sc.simplices:
        if dim < max_dim:
            for simplex in sc.simplices[dim]:
                degrees[simplex] = len(get_cofaces(sc, simplex))
    return degrees

def get_cofaces(sc, simplex):
    dim = len(simplex)-1
    out = []
    for cf in sc.simplices.get(dim+1, []):
        if set(simplex).issubset(cf):
            out.append(cf)
    return out

def compute_probability_measures(sc, degrees):
    m = {}
    max_dim = max(sc.simplices.keys())
    for dim in sc.simplices:
        if dim < max_dim:
            for simplex in sc.simplices[dim]:
                deg = degrees[simplex]
                m_F = defaultdict(float)
                for cf in get_cofaces(sc, simplex):
                    w = sc.weights[cf]
                    for face in itertools.combinations(cf, dim+1):
                        f = tuple(sorted(face))
                        if f != simplex:
                            m_F[f] += w/((dim+1)*deg)
                m[simplex] = m_F
    return m

def get_adjacent_simplices(sc, simplex):
    dim = len(simplex) - 1
    adjacent = set()
    # find simplices of the same dimension with a common face of dimension (dim - 1)
    for candidate in sc.simplices[dim]:
        if candidate != simplex and len(set(candidate) & set(simplex)) == dim:
            adjacent.add(candidate)
    return list(adjacent)


def wasserstein_distance_OT(m_F, m_G, D, reg=0.1):
    # if either distribution empty or degenerate → distance zero
    if not m_F or not m_G:
        return 0.0

    supp_F = list(m_F.keys())
    supp_G = list(m_G.keys())
    a = np.array([m_F[s] for s in supp_F], dtype=np.float64)
    b = np.array([m_G[s] for s in supp_G], dtype=np.float64)
    # normalize
    suma, sumb = a.sum(), b.sum()
    if suma > 0: a /= suma
    else:       return 0.0
    if sumb > 0: b /= sumb
    else:        return 0.0

    # cost matrix (all finite by construction)
    M = np.zeros((len(supp_F), len(supp_G)), dtype=np.float64)
    for i, u in enumerate(supp_F):
        for j, v in enumerate(supp_G):
            M[i, j] = D[(u, v)]

    # Sinkhorn
    try:
        W2 = ot.sinkhorn2(a, b, M, reg)
        return W2
    except Exception:
        # fallback to L1‐based if Sinkhorn fails
        keys = set(m_F) | set(m_G)
        l1 = sum(abs(m_F.get(k,0)-m_G.get(k,0)) for k in keys)*0.5
        return l1


def compute_ricci_curvature_OT(sc, m, D_dict, reg=1e-2):
    curvature = {}
    max_dim = max(sc.simplices.keys())
    for dim in sc.simplices:
        if dim == max_dim:
            continue
        D = D_dict[dim]         # pick the right cost map
        for simplex in sc.simplices[dim]:
            neigh = get_adjacent_simplices(sc, simplex)
            kappa = {}
            for nbr in neigh:
                W = wasserstein_distance_OT(m[simplex], m[nbr], D, reg=reg)
                kappa[nbr] = 1 - W
            curvature[simplex] = kappa
    return curvature

def update_weights(sc, curvature, delta):
    new_w = sc.weights.copy()
    max_dim = max(sc.simplices.keys())
    for dim in sc.simplices:
        if dim < max_dim:
            for s in sc.simplices[dim]:
                ks = list(curvature.get(s, {}).values())
                if not ks: continue
                avg = np.mean(ks)
                for cf in get_cofaces(sc, s):
                    new_w[cf] *= max(0, 1 - delta * avg / len(s))
    sc.weights = new_w

def identify_communities(sc):
    G = nx.Graph()
    G.add_nodes_from([v[0] for v in sc.simplices[0]])
    G.add_edges_from(sc.simplices[1])
    return list(nx.connected_components(G))

def compute_modularity(sc, communities):
    # rebuild the simple 1‐skeleton graph
    G = nx.Graph()
    G.add_nodes_from([v[0] for v in sc.simplices[0]])
    G.add_edges_from(sc.simplices[1])

    # if there are no edges, modularity is 0 by definition
    if G.number_of_edges() == 0:
        return 0.0

    # else safely compute
    return nx.algorithms.community.modularity(G, [set(c) for c in communities])


def node_labels_from_communities_dict(comms):
    labels = {}
    for i, c in enumerate(comms):
        for v in c:
            labels[v] = i
    return labels


def ricci_flow_community_detection(sc,
                                   T=10,
                                   delta=0.01,
                                   reg=1e-2,
                                   ground_truth=None,
                                   min_theta_range=0.3,
                                   theta_step=0.001):
    # precompute ground metric on (d-1)-simplices
    d = max(sc.simplices.keys()) - 1
    D = compute_ground_metric(sc, d)

    D_dict = build_all_ground_metrics(sc)

    degrees = compute_degrees(sc)
    nmi_scores, modularity_scores, ari_scores = [], [], []

    # Ricci flow loop
    for t in range(T):
        print(f"Iteration {t+1}/{T}")
        m = compute_probability_measures(sc, degrees)
        curvature = compute_ricci_curvature_OT(sc, m, D_dict, reg=reg)
        update_weights(sc, curvature, delta)

    # after flow, collect weight statistics
    max_dim = max(sc.simplices.keys())
    all_w = [w for s, w in sc.weights.items() if len(s)==max_dim]
    w_min, w_max = min(all_w), max(all_w)
    print(f"Weight range after flow: min={w_min:.3g}, max={w_max:.3g}")

    # Rescale weights to [0,1]
    all_weights = list(sc.weights.values())
    w_min, w_max = min(all_weights), max(all_weights)

    for simplex in sc.weights:
        sc.weights[simplex] = (sc.weights[simplex] - w_min) / (w_max - w_min)


    # prepare theta sweep
    theta_values = np.arange(w_min-min_theta_range, w_max, theta_step)
    orig_simp = {d: sc.simplices[d].copy() for d in sc.simplices}
    orig_w    = sc.weights.copy()

    plt.hist(list(sc.weights.values()), bins=30)
    plt.title('Weights distribution before surgery')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.show()

    for θ in theta_values:
        sc.simplices = {d: orig_simp[d].copy() for d in orig_simp}
        sc.weights   = orig_w.copy()

        # prune
        to_remove = [s for s,w in sc.weights.items() if w>θ and len(s)>1]
        for s in to_remove:
            dim = len(s)-1
            sc.simplices[dim].discard(s)
            sc.weights.pop(s, None)

        comms = identify_communities(sc)
        print(f" θ={θ:.3f} → {len(comms)} communities")

        if ground_truth:
            detected = node_labels_from_communities_dict(comms)
            common = sorted(set(ground_truth)&set(detected))
            gt = [ground_truth[v] for v in common]
            dt = [detected[v] for v in common]
            nmi_scores.append(normalized_mutual_info_score(gt, dt))
            ari_scores.append(adjusted_rand_score(gt, dt))
        else:
            nmi_scores.append(None)
            ari_scores.append(None)

        modularity_scores.append(compute_modularity(sc, comms))

    return nmi_scores, modularity_scores, theta_values, ari_scores
