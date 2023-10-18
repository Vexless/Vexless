import argparse
from cProfile import label
from k_means_constrained import KMeansConstrained
import numpy as np
import hnswlib



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


    clf = KMeansConstrained(
        n_clusters=4,
        size_min=2450000, # db1:2300000, db2:2450000 , db3: 2250000
        size_max=2550000, # db1:2700000, db2:2550000 , db3: 2750000
        random_state=0,
        init='k-means++',
        max_iter=300,
        tol=0.0001,
        verbose=False
    )

    clf.fit(vecs)


    print(vecs.shape)


    # print((clf.cluster_centers_))

    import matplotlib.pyplot as plt
    labels = clf.labels_
    print(labels)
    unique, counts = np.unique(labels, return_counts=True)

    for cluster_id, count in zip(unique, counts):
        print(f"Cluster {cluster_id} has {count} points")

    count_arr = np.bincount(labels)


    # Generate the partitions list dynamically
    partitions = ['P{}'.format(i) for i in range(clf.n_clusters)]

    # Create a bar chart
    plt.bar(partitions, count_arr)

    # Add labels and title
    plt.xlabel('Partitions')
    plt.ylabel('Number of Data Elements')
    plt.ylim([0,3000000])
    plt.title('Number of Data Elements in Each Partition')

    # Display the chart
    plt.savefig(res_save+'constrained_clustered.png')


    ## index building!
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
    for idx in range(len(labels)):
        if idx%(one_percent_of_dbsize)==0:
            print("%% %d indexing finished!"%int(idx/one_percent_of_dbsize))
        if labels[idx] == [0]:
            dic0_id_2_base.append(idx)
            c0+=1
            p0.add_items( vecs[idx] )
        elif labels[idx] == [1]:
            dic1_id_2_base.append(idx)
            c1+=1
            p1.add_items( vecs[idx] )
        elif labels[idx] == [2]:
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

  
    # localize the id and distance maps to centroids.
    np.savetxt(res_save+"Deep1M_faiss_kmeans_4partitions_Ids_belong_which_centroids.csv", labels.astype(int), delimiter=",")
    np.save(res_save+"Deep1M_faiss_kmeans_4partitions_centroids.npy", clf.cluster_centers_)

    # localize the distributed indexes
    index_path0=res_save+'faiss_kmeans_L2_index_Deep1Mef200m16_0.bin'
    index_path1=res_save+'faiss_kmeans_L2_index_Deep1Mef200m16_1.bin'
    index_path2=res_save+'faiss_kmeans_L2_index_Deep1Mef200m16_2.bin'
    index_path3=res_save+'faiss_kmeans_L2_index_Deep1Mef200m16_3.bin'


    p0.save_index(index_path0)
    p1.save_index(index_path1)
    p2.save_index(index_path2)
    p3.save_index(index_path3)


    # # localize the map of to original index (for evaluation with ground truth)
    # with open(res_save+'partition0_2_base_L2_idx.txt', 'w') as fp:
    #     for item in dic0_id_2_base:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print('localized p0 to '+'/home/su311/deep/index/clustered_index/balanced_deep10M/partition0_2_base_L2_idx.txt')


    # with open(res_save+'partition1_2_base_L2_idx.txt', 'w') as fp:
    #     for item in dic1_id_2_base:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print('localized p1 to '+'/home/su311/deep/index/clustered_index/balanced_deep10M/partition1_2_base_L2_idx.txt')


    # with open(res_save+'partition2_2_base_L2_idx.txt', 'w') as fp:
    #     for item in dic2_id_2_base:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print('localized p2 to '+'/home/su311/deep/index/clustered_index/balanced_deep10M/partition2_2_base_L2_idx.txt')

    # with open(res_save+'partition3_2_base_L2_idx.txt', 'w') as fp:
    #     for item in dic3_id_2_base:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print('localized p3 to '+'/home/su311/deep/index/clustered_index/balanced_deep10M/partition3_2_base_L2_idx.txt')
    
