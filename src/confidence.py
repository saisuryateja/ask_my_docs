def is_confident(distances, abs_threshold=1.1, gap_threshold=0.02):
    """
    distances: list or array of FAISS L2 distances (lower is better)
    """

    if len(distances) == 0:
        return False

    # Gate A: absolute confidence
    if distances[0] > abs_threshold:
        return False

    # Gate B: relative confidence (ambiguity check)
    if len(distances) > 1:
        if abs(distances[1] - distances[0]) < gap_threshold:
            return False

    return True
