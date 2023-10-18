from math import ceil, floor
import numpy as np
import argparse
import faiss



def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
 
def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file")
    parser.add_argument("--dst", help="The output file path")
    parser.add_argument("--topk", type=int, help="The number of element to pick up")
    parser.add_argument("--k", type=int, help="The number of desired clusters.")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = process_args()
    db_size = int(args.topk)
    ncentroids = int(args.k)
    res_save = args.dst
    base_fname = args.src
    

    vecs = read_fbin(base_fname, chunk_size=db_size)
    print("vecs.shape: ",vecs.shape)


    ncentroids = 8
    niter = 100 # adjustable
    verbose = True
    d = vecs.shape[1] # d
    min_points_per_centroid = floor(0.9*db_size/ncentroids) # 0.9 is an example, you may adjust accordingly
    max_points_per_centroid = ceil(1.1*db_size/ncentroids) # 1.1 is an example, you may adjust accordingly
    print("min_points_per_centroid, max_points_per_centroid = ", min_points_per_centroid, max_points_per_centroid)
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose,
    min_points_per_centroid=min_points_per_centroid,max_points_per_centroid=max_points_per_centroid)

    kmeans.train(vecs)

    # print("kmeans.centroids: ",kmeans.centroids)
    # print("kmeans.k: ",kmeans.k)
    # print("kmeans.cp: ",kmeans.cp)
    # print("kmeans.d: ",kmeans.d)
    # print("kmeans.index: ",kmeans.index)
    # print("kmeans.obj: ",kmeans.obj)

    D, I = kmeans.index.search(vecs, 1)

    # Cluster_count
    cluster_counts = np.bincount(I.flatten())

    # Create a dictionary to store the number of vectors in each partition
    partition_counts = {}
    for idx, count in enumerate(cluster_counts):
        partition_name = f"Partition{idx+1}"
        partition_counts[partition_name] = count

    print(partition_counts)
    
    # localize the id and distance maps to centroids.
    np.savetxt(res_save+"I.csv", I.astype(int), delimiter=",")
    np.savetxt(res_save+"D.csv", D, delimiter=",")