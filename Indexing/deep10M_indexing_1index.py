from math import ceil, floor
import numpy as np
import argparse
import struct
import faiss
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
    
    vecs = read_fbin(base_fname, chunk_size=db_size)
    print("vecs.shape: ",vecs.shape)

    c0 = 0
    
    p0 = hnswlib.Index(space='l2', dim=vecs.shape[1])  # possible options are l2, cosine or ip
    p0.init_index(max_elements=db_size, ef_construction=200, M=16)
    # p0.set_num_threads(1)


    # index 1
    one_percent_of_dbsize = int(db_size/100)
    for idx in range(db_size):
        if idx%(one_percent_of_dbsize)==0:
            print("%% %d indexing finished!"%int(idx/one_percent_of_dbsize))
        c0+=1
        p0.add_items( vecs[idx] )
    index_path0='deep{db_size}_l2_efc200_m16.bin'
    
    p0.save_index(index_path0)
    print(c0)

    