def is_confident(distances, abs_threshold=1.1):
    """
    distances: list or array of FAISS L2 distances (lower is better)
    """

    if len(distances) == 0:
        return False

    # Gate A: absolute confidence (is the best match good enough?)
    if distances[0] > abs_threshold:
        return False

    return True
