from math import ceil, floor
import numpy as np
import argparse
import struct
import faiss
import hnswlib

db_size = 10**7

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
    parser.add_argument("--src", help="The input file (.ibin)")
    parser.add_argument("--dst", help="The output file (.bin)")
    parser.add_argument("--topk", type=int, help="The number of element to pick up")
    return parser.parse_args()

if __name__ == "__main__":
    
    deep10M_base_fname = '/home/su311/deep/data/deep10M/base/base.10M.fbin'
    vecs = read_fbin(deep10M_base_fname, chunk_size=db_size)
    print("vecs.shape: ",vecs.shape)


    ncentroids = 4
    niter = 100
    verbose = True
    d = vecs.shape[1] # d
    min_points_per_centroid = floor(0.9*db_size/ncentroids)
    max_points_per_centroid = ceil(1.1*db_size/ncentroids)
    print("min_points_per_centroid, max_points_per_centroid = ", min_points_per_centroid, max_points_per_centroid)
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose,
    min_points_per_centroid=min_points_per_centroid,max_points_per_centroid=max_points_per_centroid)

    kmeans.train(vecs)

    print("kmeans.centroids: ",kmeans.centroids)
    print("kmeans.k: ",kmeans.k)
    print("kmeans.cp: ",kmeans.cp)
    print("kmeans.d: ",kmeans.d)
    print("kmeans.index: ",kmeans.index)
    print("kmeans.obj: ",kmeans.obj)

    D, I = kmeans.index.search(vecs, 1)
    
    cluster_counts = np.bincount(I.flatten())
    # Create a dictionary to store the number of vectors in each partition
    partition_counts = {}
    for idx, count in enumerate(cluster_counts):
        partition_name = f"Partition{idx+1}"
        partition_counts[partition_name] = count

    print(partition_counts)
    
    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0

    p0 = hnswlib.Index(space='l2', dim=vecs.shape[1])  # possible options are l2, cosine or ip
    p0.init_index(max_elements=int(db_size/2), ef_construction=200, M=16)
    dic0_id_2_base = []
    p1 = hnswlib.Index(space='l2', dim=vecs.shape[1])  # possible options are l2, cosine or ip
    p1.init_index(max_elements=int(db_size/2), ef_construction=200, M=16)
    dic1_id_2_base = []
    p2 = hnswlib.Index(space='l2', dim=vecs.shape[1])  # possible options are l2, cosine or ip
    p2.init_index(max_elements=int(db_size/2), ef_construction=200, M=16)
    dic2_id_2_base = []
    p3 = hnswlib.Index(space='l2', dim=vecs.shape[1])  # possible options are l2, cosine or ip
    p3.init_index(max_elements=int(db_size/2), ef_construction=200, M=16)
    dic3_id_2_base = []
    
    
    one_percent_of_dbsize = int(db_size/100)    
    for idx in range(len(I)):
        if idx%(one_percent_of_dbsize)==0:
            print("%% %d indexing finished!"%int(idx/one_percent_of_dbsize))
        if I[idx] == [0]:
            dic0_id_2_base.append(idx)
            c0+=1
            p0.add_items( vecs[idx] )
        elif I[idx] == [1]:
            dic1_id_2_base.append(idx)
            c1+=1
            p1.add_items( vecs[idx] )
        elif I[idx] == [2]:
            dic2_id_2_base.append(idx)
            c2+=1
            p2.add_items( vecs[idx] )
        else:
            dic3_id_2_base.append(idx)
            c3+=1
            p3.add_items( vecs[idx] )

    print(f"We have {c0} in partition 0.")
    print(f"We have {c1} in partition 1.")
    print(f"We have {c2} in partition 2.")
    print(f"We have {c3} in partition 3.")

    index_path = '/home/su311/deep/index/clustered_index/deep10M/'
    # localize the id and distance maps to centroids.
    np.savetxt(index_path+"Deep10M_faiss_kmeans_4partitions_Ids_belong_which_centroids.csv", I.astype(int), delimiter=",")
    np.savetxt(index_path+"Deep10M_faiss_kmeans_4partitions_Ids_Distance_to_closest_centroids.csv", D, delimiter=",")
    np.save(index_path+"Deep10M_faiss_kmeans_4partitions_centroids.npy", kmeans.centroids)

    # localize the distributed indexes
    index_path0=index_path+'faiss_kmeans_L2_index_Deep10Mef200m16_0.bin'
    index_path1=index_path+'faiss_kmeans_L2_index_Deep10Mef200m16_1.bin'
    index_path2=index_path+'faiss_kmeans_L2_index_Deep10Mef200m16_2.bin'
    index_path3=index_path+'faiss_kmeans_L2_index_Deep10Mef200m16_3.bin'
  

    p0.save_index(index_path0)
    p1.save_index(index_path1)
    p2.save_index(index_path2)
    p3.save_index(index_path3)


    # localize the map of to original index (for evaluation with ground truth)
    with open(r'/home/su311/deep/index/clustered_index/deep10M/partition0_2_base_L2_idx.txt', 'w') as fp:
        for item in dic0_id_2_base:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('localized p0 to '+'/home/su311/deep/index/clustered_index/deep10M/partition0_2_base_L2_idx.txt')


    with open(r'/home/su311/deep/index/clustered_index/deep10M/partition1_2_base_L2_idx.txt', 'w') as fp:
        for item in dic1_id_2_base:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('localized p1 to '+'/home/su311/deep/index/clustered_index/deep10M/partition1_2_base_L2_idx.txt')


    with open(r'/home/su311/deep/index/clustered_index/deep10M/partition2_2_base_L2_idx.txt', 'w') as fp:
        for item in dic2_id_2_base:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('localized p2 to '+'/home/su311/deep/index/clustered_index/deep10M/partition2_2_base_L2_idx.txt')

    with open(r'/home/su311/deep/index/clustered_index/deep10M/partition3_2_base_L2_idx.txt', 'w') as fp:
        for item in dic3_id_2_base:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('localized p3 to '+'/home/su311/deep/index/clustered_index/deep10M/partition3_2_base_L2_idx.txt')