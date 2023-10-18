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


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]
    

def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def fvecs_read(filename, topK):
    vecs = []    
    with open(filename, "rb") as f:
        while 1:
            # The first 4 byte is for the dimensionality
            dim_bin = f.read(4)
            if dim_bin == b'':
                break

            # The next 4 * dim byte is for a vector
            dim, = struct.unpack('i', dim_bin)
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            
            # Store it
            vecs.append(vec)
            if len(vecs) == topK:
                break
    vecs = np.array(vecs, dtype=np.float32)
    assert vecs.shape[0] == topK
    return vecs


def ivecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.ibin)")
    parser.add_argument("--dst", help="The output file (.bin)")
    parser.add_argument("--topk", type=int, help="The number of element to pick up")
    return parser.parse_args()



if __name__ == "__main__":
    args = process_args()
    db_size = int(args.topk)
    ncentroids = int(args.k)
    res_save = args.dst
    base_fname = args.src
    vecs = read_fbin(base_fname, chunk_size=db_size)
    print("vecs.shape: ",vecs.shape)
    vecs = fvecs_read(base_fname, 10**6)

    print("vecs.shape: ",vecs.shape)

    c0 = 0
    p0 = hnswlib.Index(space='l2', dim=960)  # possible options are l2, cosine or ip
    p0.init_index(max_elements=10**6, ef_construction=500, M=24)
    # p0.set_num_threads(1)


    # index 1
    for idx in range(10**6):
        if idx%10000==0:
            print("%% %d indexing finished!"%int(idx/10000))
        c0+=1
        p0.add_items( vecs[idx] )
    
    print(c0)


    index_path0=res_save+'index_GIST1M_ef500_M24.bin'
    
    p0.save_index(index_path0)