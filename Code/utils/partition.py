def generate_singleton(vertices):
    return [[v] for v in vertices]

def generate_partition(vertices, k):
    """
    頂点集合をk個の部分集合に分割する

    """
    n = len(vertices)
    if k > n:
        raise ValueError("k must be less than or equal to the number of vertices.")
    
    res = []
    cnt = 0
    for i in vertices:
        if cnt < k:
            res.append([i])
            cnt += 1
        else:
            res[cnt % k].append(i)
    
    return res