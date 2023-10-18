# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
from time import time
import numpy as np
import azure.functions as func
from azure.storage.blob import ContainerClient
from io import BytesIO, StringIO
import subprocess


import hnswlib

def main(name: str) -> str:
    index_fname = 'Your index_fname'
    conn_str = "Your account conn str"
    blob_container_name = "Your blob_container_name"
    query_fname = 'query file name'
    ground_truth_fname = 'gt file name'
    index_fname = 'Your index_fname'
    
    prepare_start_time = time()
    container_client = ContainerClient.from_connection_string(
        conn_str=conn_str, 
        container_name=blob_container_name
    )
 
    index = container_client.download_blob(index_fname)

    query = container_client.download_blob(query_fname)
    
    
    filepath = 'path2index'
    index_local_Bytes = BytesIO(index.content_as_bytes())
    with open(filepath, 'wb') as f:
        f.write(index_local_Bytes.getbuffer())
    f.close()


    query_filepath = 'path2query'
    index_local_Bytes = BytesIO(query.content_as_bytes())
    with open(query_filepath, 'wb') as f:
        f.write(index_local_Bytes.getbuffer())
    f.close()


    def mmap_bvecs(fname):
        x = np.memmap(fname, dtype='uint8', mode='r')
        d = x[:4].view('int32')[0]
        return x.reshape(-1, d + 4)[:, 4:]

    def fvecs_read(filename, c_contiguous=True):
        fv = np.fromfile(filename, dtype=np.float32)
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


    
    
    query = mmap_bvecs(query_filepath)
    # gt = ivecs_read(gt_filepath)
    QUERY_NUM = query.shape[0]
    
    prepare_end_time = time()
    
    # LOAD index
    index_load_start_time = time()
    DIM = 128
    p = hnswlib.Index(space='l2', dim=DIM)  # possible options are l2, cosine or ip
    p.load_index(filepath)
    index_load_end_time = time()
    

    
    EFs = [] 
    res = []
    for EF in EFs:

        p.set_ef(EF)
        
        def query_func(K):

            # QUERY:
            query_10k_time = []
            for q in query:
                start_time = time()
                # labels_q, distances_q = p.knn_query(q, K)
                p.knn_query(q, K)
                end_time = time()
                query_10k_time.append(end_time - start_time) 

            # Measure recall
            correct = 0

            return float(correct)/(K*QUERY_NUM) , query_10k_time
        # load index on cloud

        
        # set search scope, find the topK
        p95_latency = 0
        min_time = 9999

        for j in range(1):
            r,t = query_func(1)
            p95_latency = np.percentile(t, 95)
            # min_time = min(t)
            per_query_time = sum(t)/len(t)

        res.append(str(EF)+'    '+str(r)+'    '+str(round(per_query_time,2))+'s')

    logging.info('Python HTTP trigger function processed a request.')

    avx_info = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode('utf-8')
    simd = False
    if avx_info.find('avx2') != -1:
        simd = True
    
    return f"Hello , index name: {index_fname}, avg query: {per_query_time}s + {avx_info[79:120]}. + {str(simd)} + QUERY_NUM: {str(QUERY_NUM)} + index load time = {str(index_load_end_time-index_load_start_time)}s, query 95th tail time = {str(p95_latency)}s, prepare time = {str(prepare_end_time-prepare_start_time)}"
    return f"QUERY_NUM: {str(QUERY_NUM)} + index load time = {str(index_load_end_time-index_load_start_time)}s, query 95th tail time = {str(0)}s, prepare time = {str(prepare_end_time-prepare_start_time)}"

    # return f"Hello {name}!"
