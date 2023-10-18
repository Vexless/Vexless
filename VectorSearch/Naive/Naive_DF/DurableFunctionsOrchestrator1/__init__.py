# This function is not intended to be invoked directly. Instead it will be
# triggered by an HTTP starter function.
# Before running this sample, please:
# - create a Durable activity function (default name is "Hello")
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

from gettext import npgettext
from http.client import CONTINUE
import logging
import json
import azure.functions as func
import azure.durable_functions as df
import subprocess
import numpy as np
import io
import contextlib


def orchestrator_function(context: df.DurableOrchestrationContext):
    partition_cnt = 5 # you can specify here
    # work_batch = list(range(partition_cnt + 1)) # partition id
    
    Ts = 0.95
    ## determine the partition ID:
    ## received from external query handler that you can specify on Azure portal, multiple source is supported.
    cache = {} # the caching hashmap active in orchestrator.
    res = [] # the query result.
    centroids = np.loadtxt('/Path/of/centroids')
    # Listen for events indefinitely:
    while True:
        # This will wait for an event named "NewQueryVector" to be raised to this instance

        query = yield context.wait_for_external_event('Query')
        vec_id_val = cache.get(query)

        # Check if the value exists
        if vec_id_val is not None:
            res.append(vec_id_val)
            continue
        else: ## cache miss
            d_partition_cnt = [0]*partition_cnt
            work_batch = []
            for d in range(partition_cnt):
                d_partition_cnt[d]= np.linalg.norm(query - centroids[d])
                if d_partition_cnt[d]<=Ts:
                    work_batch.append(d)
            
            parallel_tasks = [ context.call_activity("Functions", b) for b in work_batch ] # specifying DEF ids.
            for i, task in enumerate(parallel_tasks):
                logging.info(f"Parallel activity function #{i + 1} activated.")
            start_time = context.current_utc_datetime
            outputs = yield context.task_all(parallel_tasks)
            # outputs = yield context.task_all(times)
            end_time =  context.current_utc_datetime
            execution_time = end_time - start_time
            
            # Create a text buffer to capture the output
            output_buffer = io.StringIO()

            # Redirect the standard output to the buffer
            with contextlib.redirect_stdout(output_buffer):
                np.__config__.show()

            # Get the content of the buffer as a string
            config_info = output_buffer.getvalue()

            ## caching
            # Get the (id, distance_score) tuple with the minimum distance_score, and extract the id from the tuple
            min_distance_tuple = min(outputs, key=lambda x: x[1])
            min_distance_id = min_distance_tuple[0]
            cache[query] = min_distance_id 
            tmp_res =  {
                # "NumPy":config_info,
                # "AVX":subprocess.check_output("cat /proc/cpuinfo", shell=True).decode('utf-8'),
                "query":query,
                "results": min_distance_id,  # Replace this with the actual results of your orchestrator logic
                "execution_time": execution_time.total_seconds()
            }
            res.append(tmp_res)
    return res

main = df.Orchestrator.create(orchestrator_function)