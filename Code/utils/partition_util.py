import random

def generate_random_partition(vertices, num_partitions):
    if num_partitions > len(vertices):
        raise ValueError("Number of partitions cannot exceed the number of vertices.")
    
    shuffled_vertices = vertices[:]
    random.shuffle(shuffled_vertices)
    
    partitions = [[] for _ in range(num_partitions)]
    for i, vertex in enumerate(shuffled_vertices):
        partitions[i % num_partitions].append(vertex)
    
    return partitions

def generate_unique_partitions(vertices, num_partitions, num_samples):
    unique_partitions = set()
    
    while len(unique_partitions) < num_samples:
        partition = generate_random_partition(vertices, num_partitions)
        
        partition_tuple = tuple(tuple(sorted(community)) for community in partition)
        
        unique_partitions.add(partition_tuple)
    
    unique_partitions = [list(map(list, partition)) for partition in unique_partitions]
    
    return unique_partitions

if __name__ == "__main__":
    vertices = list(range(10))
    num_partitions = 2
    
    partition = generate_random_partition(vertices, num_partitions)
    print("Vertices:", vertices)
    print("Generated Partition:", partition)
