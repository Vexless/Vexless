from time import time
import hnswlib
import numpy as np
# acesss the index from storage

# Download the index from the storage
DEST_FILE = '/path/to/file/'
# with open(DEST_FILE, "wb") as my_blob:
#     download_stream = blob_client.download_blob()
#     my_blob.write(download_stream.readall())

# get index path
DIM = 96
index_path0 = DEST_FILE+'index_file.bin'


idx_PATH = '/path/to/file/'
# SIFT1M_CSV_PATH = blob_url.split('//')[-2]
# LOAD DATA
def read_vector_from_csv(vector_numpy_set_str_name):
    return np.loadtxt(idx_PATH+vector_numpy_set_str_name+'.csv', delimiter=",")
query = read_vector_from_csv('query')
# base = read_vector_from_csv('base')
gt = read_vector_from_csv('gt')
print(gt.shape, query.shape)
# VEC_NUM = base.shape[0]
QUERY_NUM = query.shape[0]

# K = gt.shape[1]

# LOAD index
start_time = time()
p0 = hnswlib.Index(space='l2', dim=DIM)  # possible options are l2, cosine or ip
p0.load_index(index_path0)
p0.set_num_threads(1)


end_time = time()
EFs = [1,2,3,4,5,8,10,20,30,50,80,120,160]

print(f"INDEX LOADED FROM"+str(index_path0)+", AND TOOK: %f",end_time-start_time)

## Set different EF_Search
for EF in EFs:
    print(f"---------- EF  ==  %d ----------",EF)
    p0.set_ef(EF)
    # query function with recall and query time return:
    def query_func(K):

        # QUERY:
        global_id_in_query = []
        start_time = time()
        total_q_time = 0.0
        for q in query:
            q_start_time = time()
            ids_per_q_0, distances_q_0 = p0.knn_query(q, K)
            q_end_time = time()
            global_id_in_query.append(ids_per_q_0[0])
            total_q_time += q_end_time-q_start_time
            
        end_time = time()

        # Measure recall
        correct = 0
        for i in range(QUERY_NUM):
            for label in global_id_in_query[i]:
                if label in gt[i][:K]:
                    correct += 1

        return float(correct)/(K*QUERY_NUM) , end_time - start_time, total_q_time
    # load index on cloud
    print("TIME/ RECALL\t ACTUAL QUERY TIME :")
    for i in [1]:
        sum_time = 0.0
        sum_q_time = 0
        min_time = 9999
        min_q_time = 99999
        for j in range(10):
            r,t,q_t = query_func(i)
            sum_time += t
            sum_q_time += q_t
            
        avg_time = sum_time/10
        avg_q_time = sum_q_time/10

        print(str(avg_time)+'/'+str(r)+"\t"+str(avg_q_time))
        
