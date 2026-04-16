from itertools import combinations

def get_pairs_with_singletons(clusters):
    """
    Return pairs for clusters, but also treat singletons as correct units.
    Each pair is a frozenset. Singleton items are represented as frozenset of size 1.
    """
    pairs = set()
    for cluster in clusters:
        if len(cluster) == 1:
            pairs.add(frozenset(cluster))  # singleton counts as a "pair"
        else:
            for pair in combinations(cluster, 2):
                pairs.add(frozenset(pair))
    return pairs

def pairwise_precision_recall_f1_singletons(gt_clusters, pred_clusters):
    """
    Compute precision, recall, F1 considering singletons.
    """
    gt_pairs = get_pairs_with_singletons(gt_clusters)
    pred_pairs = get_pairs_with_singletons(pred_clusters)

    true_positives = gt_pairs & pred_pairs
    precision = len(true_positives) / len(pred_pairs) if pred_pairs else 0
    recall = len(true_positives) / len(gt_pairs) if gt_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# ---------------- Sample Example ----------------
gt = [['a'], ['b'], ['c', 'd']]
pred1 = [['b'], ['a'], ['d', 'c']]      # perfect match
pred2 = [['a', 'b', 'c'], ['d']]        # merged clusters
pred3 = [['a'], ['b'], ['c'], ['d']]    # all split
pred4 = [['a', 'b',], ['c', 'd']]        # merged clusters

for i, pred in enumerate([pred1, pred2, pred3, pred4], 1):
    precision, recall, f1 = pairwise_precision_recall_f1_singletons(gt, pred)
    print(f"Prediction {i}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")