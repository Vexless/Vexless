from common import *
        
import time
import random
from collections.abc import Iterable, Callable
import scipy.signal

help_str = '''Parameters:
  o: outer distribution, can be gaussian, zipf or uniform (default: zipf)
  a1 (mu): (outer/low-pass) skewness for zipf, standard deviation for gaussian, discarded
  a2 (a): (inner/high-pass) skewness, 
  g: granularity, number of bins.(default: 10)
  n: number of samples
  duration: range of samples (in seconds)
  offset: offset of samples (delay in seconds)
  f: save/load file location (default: ./plan.bin)
  mode: 
       get_distribution: get and save the distribution
       generate: generate queries based on the distribution
'''


parameters = parameters_t()
plan = None
n_threads = None
console_debug = lambda *_, **__: None

def init(seed = 1):
    global parameters
    import os
    if os.path.exists('seeds'):
        with open('seeds', 'r') as fp:
            try:
                seeds = tuple(int(s.strip()) for s in fp.read().split(' ') if s.strip())
                np.random.seed(seeds[0])
                random.seed(seeds[1])
                parameters.seeds = seeds
                print(f'Using random seeds {seeds}.')
                return
            except: pass
    random.seed(time.time() + seed)
    npseed = int(random.random() * time.perf_counter_ns()) % uint32_max
    np.random.seed(npseed)
    randomseed = int(np.random.random() * time.perf_counter_ns()) 
    random.seed(randomseed)
    parameters.seeds = (npseed, randomseed) # save the seeds for reproducibility

def controlled_shuffle(data, max_shift):
    n = len(data)
    if n < 3: return
    max_shift = int(min(max_shift, (n-1) // 2 - 1))
    for i in random.sample(range(1, (n-1)//2), max_shift):
        ss = data[:i]
        data[:i] = data[-i:]
        data[-i:] = ss
        
def gen_distribution():
    global plan
    def get_normalized(a, b, n, r, postporcess, 
                       distribution = distribution_f[distribution_t.zipf]):
        if n == 0: return np.empty(0)
        data = distribution(a, b, n).astype(np.float64)
        data = postporcess(data, b)
        data *= r
        return data
    
    def postprocess_outer(data, b):
        data /= np.max(data)
        data = data *(1 - b) + b * (np.mean(data)+truncnorm_01rand(1, len(data))*.1) 
        return data / np.sum(data)
    weights = get_normalized(parameters.a1, 
                             parameters.b1, 
                             parameters.granularity, 
                             parameters.n, 
                             postprocess_outer, 
                             distribution = distribution_f[parameters.outer]
            )
    if parameters.s1 > 1:
        np.random.shuffle(weights)
    else:
        controlled_shuffle(weights, (len(weights) - 1 // 2 - 1) * parameters.s1)
    weights = np.round(weights).astype(np.int32)
    weights[-1] = max(parameters.n - np.sum(weights[:-1]), 0)
    console_debug(f'weights: \n{weights}')
    offsets = (np.array(range(0, parameters.granularity), dtype=np.float64)/parameters.granularity) * parameters.duration
    duration_per_segment = parameters.duration / parameters.granularity
    def postprocess(data, b):
        n = len(data)
        sample_size = int(n * (1 - b))
        if sample_size > 0:
            data = random.sample(list(data), sample_size)
        elif sample_size <= 0:
            data = []
        if (sample_size > n):
            print(f'Warning: sample size {sample_size} < n {n}')
        print(sample_size)
        if parameters.b3 > 0:
            window = int(parameters.b3)
            window = np.ones(window) / window
            data = np.convolve(data, window, mode = 'valid')
        if data:
            data = data / np.max(data)  
        data = np.append(data, (np.linspace(0, 1, n - sample_size) + truncnorm_01rand(1, n - sample_size)*parameters.b4))
        # data = np.sort(data)
        # if parameters.shfl > 0 and parameters.shfl <= 1:
        #     controlled_shuffle(data, (len(data) - 1 // 2 - 1) * parameters.shfl)
        # elif parameters.shfl > 1:
        # np.add.accumulate(data, out = data)
        return np.sort(data)
    
    plan = [k for w, off in zip(weights, offsets) for k in 
            get_normalized(parameters.a2, 
                           parameters.b2,
                           w, 
                           duration_per_segment, 
                           postprocess,
                           distribution = distribution_f[parameters.inner]
    ) + off]
    
    print(len(plan) - parameters.n)
    # console_debug(f'weights: \n{plan}')
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    # plan = savgol_filter(plan, window_length=10, polyorder=4)
    # plan = np.convolve(plan, 15, mode = 'full')
    # plan = gaussian_filter1d(plan, 10)
    with open(parameters.f, 'wb') as fp:
        pickle.dump(dump_t(parameters, plan), fp)

def generate_impl(workload : Iterable[Callable]): # submit workload functions as a list of functions
    time.sleep(parameters.offset) # sleep off the offset
    t0 = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        while len(plan) > 0:
            t = plan.pop(0)
            while t > time.perf_counter() - t0:
                if t < time.perf_counter() - t0 + .03: # python's sleep is not accurate enough
                    while t > time.perf_counter() - t0: # when delta_t < epsilon, busy wait
                        continue
                    else:
                        executor.submit(workload.pop(0), t) # submit the workload precisely at time t.
                        break
                else: 
                    time.sleep(t - (time.perf_counter() - t0) - .03)

def generate(workload : Iterable[Callable], plan_path: str):
    with open(plan_path, 'rb') as fp:
        dump : Optional[dump_t] = pickle.load(fp)
        global plan, parameters
        plan = dump.plan
        parameters = dump.parameters
        generate_impl(workload)
        
def main():
    import sys, copy
    init()
    argv = copy.deepcopy(sys.argv)
    global parameters, console_debug
    
    print_help = False
    argv.pop(0)

    def get_distribution(name: str):
        nonlocal argv
        d = 'zipf'
        match argv.pop(0).lower(): 
            case a if a.startswith('g') or a.startswith('n'):
                d = 'normal'
            case a if a.startswith('z'):
                d = 'zipf'
            case a if a.startswith('u'):
                d = 'uniform'
            case a if a.startswith('p'):
                d = 'poisson'
            case a if a.startswith('i'):
                d = 'inv_gaussian'
            case _:
                return
        exec(f'parameters.{name} = distribution_t.{d}')

    while len(argv) > 0:
        arg = argv.pop(0)
        match arg.lower().strip():
            case '-o' | '--outer':
                get_distribution('outer')
            case '-i' | '--inner':
                get_distribution('inner')
            case '-a1' | '--a1' | '-mu' | '--mu':
                try: parameters.a1 = float(argv.pop(0))
                except Exception as e: print(e)
            case '-a2' | '--a2' | '-a' | '--a':
                try: parameters.a2 = float(argv.pop(0))
                except Exception as e: print(e)
            case '-b1' | '--b1':
                try: 
                    b1 = float(argv.pop(0))
                    if b1 >= 0 and b1 <= 1: 
                        parameters.b1 = b1
                except Exception as e: print(e)
            case '-b2' | '--b2':
                try: 
                    b2 = float(argv.pop(0))
                    if b2 >= 0 and b2 <= 1: 
                        parameters.b2 = b2
                except Exception as e: print(e)
            case '-b3' | '--b3':
                try: 
                    b3 = float(argv.pop(0))
                    if b3 > 0 : 
                        parameters.b3 = b3
                except Exception as e: print(e)
            case '-b4' | '--b4':
                try: 
                    b4 = float(argv.pop(0))
                    if b4 > 0 : 
                        parameters.b4 = b4
                except Exception as e: print(e)
            case '-n' | '--n':
                try: parameters.n = int(argv.pop(0))
                except Exception as e: print(e)
            case '-d' | '--duration':
                try: parameters.duration = int(argv.pop(0))
                except Exception as e: print(e)
            case '-o' | '--offset': 
                try: parameters.offset = float(argv.pop(0))
                except Exception as e: print(e)
            case '-g' | '--granularity':
                try: parameters.granularity = int(argv.pop(0))
                except Exception as e: print(e)
            case '-s1' :
                try: 
                    shfl = float(argv.pop(0))
                    if shfl >= 0 : 
                        parameters.s1 = shfl
                except Exception as e: print(e)
            case '-s' | '--shuffle':
                try: 
                    shfl = float(argv.pop(0))
                    if shfl >= 0: 
                        parameters.shfl = shfl
                except Exception as e: print(e)
            case '-m' | '--mode':
                try: 
                    match argv.pop(0).lower().strip()[:3]:
                        case 'get' | '0': 
                            parameters.mode = mode_t.get_distribution
                        case 'exec' | 'issue' | 'gen' | '1':
                            parameters.mode = mode_t.generate
                        case s:
                            raise ValueError(f'Invalid mode {s}')
                except Exception as e: print(e)
            case '-f' | '--f':
                try: parameters.f = argv.pop(0)
                except Exception as e: print(e)
            case '-h' | '--help':
                print_help = True
            case '-v' | '--verbose':
                console_debug = print
            case s:
                print(f'Invalid option: {s}')
                print_help = True
    if print_help: 
        print(help_str)
        
    console_log('Parameters:')
    for k, v in parameters.__dict__.items():
        console_log(f'    {k}: {v}')
    if parameters.mode == mode_t.get_distribution:
        gen_distribution()
    elif parameters.mode == mode_t.generate:
        with open(parameters.f, 'rb') as fp:
            dump : Optional[dump_t] = pickle.load(fp)
            global plan
            plan = dump.plan
            parameters = dump.parameters
            generate([lambda: console_log(f'executing {i}') for i in range(parameters.n)])
            
if __name__ == "__main__":
    main()
