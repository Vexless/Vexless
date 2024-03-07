# Bursty workload generator
## How it works
**Workload Generation `main.py -m get`**: This script will first split workloads into *g* bins using the *outer* distribution (default is zipf, can also be gaussian, poisson, or uniform). Each bin will get $w_k$ workloads and $\Sigma_{k=0}^{g}w_k = n$. Each bin will start at $offset + \frac{duration}{g}*k$ second. The inner distribution will determine the time each workload query will be issued such that $\omega_{k, i} \sim \zeta(a_2, 1)$. $\omega_{k, i}$ is then scaled to the window of the k-th bin: $w_{k, i} = offset + (\omega_{k, i} + k)*s$, where $s = \frac{duration}{g}$. Consider using uniform distribution for less extreme distribution of works per bin.

**Visualization**: Use vis_plan.py to visualize the workload plan located in `./plan.bin` using a histogram.

**Executing workload `main.py -m exec`**: Issuing the workload using the plan in `./plan.bin`. The workloads being issued should be provided in a queue (list) of callables.

## Parameters (main.py):
  - **-g**: outer distribution (default zipf) can be gaussian (`-g gauss`) or uniform (`-g uniform`)
  - **$a_1$(--a1) or $\mu$(--mu)**: (outer/low-pass) skewness, aka. how even are workload spread across groups, lower more skewed $a_1\to1$: most skewed, $a_1\to+\infty$: uniform. For gaussian distribution as outer distribution, this is the std ($\mu$).
  - **$a_2$(--a2) or --a**: (inner/high-pass) skewness, aka. how skewed data is within each group
  - **n**: number of samples
  - **granularity**: number of groups/bins (big spikes if a1 is low)
  - **duration**: range of samples (in seconds)
  - **offset**: offset of samples (delay in seconds)
  - **f**: save/load file location (default: ./plan.bin)
  - **mode**: 
      - get_distribution: get and save the distribution (default)
      - generate: test workload generation with dummy workload

## Visualizing result (vis_plan.py):
  **Interactive Visualizer** `vis_plan.py -i`:    Automatically activated when attached to a debugger. Users can generate, visualize, manipulate, and save workloads in an interactive prompt, which makes it easier to design workloads.
  - **Any** input not specified in this section will be passed to main.py. For example, `-o n -i z -n 10000` will launch `main.py -o n -i z -n 10000`.
  - *Commands*: 
    - `load`: Load workload form `./plan.bin`
    - `show`: Show visualization of the current workload.
    - `save`: Save visualization and workload + parameters in `./saved_plans`.
    - `scale [scale: float] [(optional)shift: float]`: Scale the workload.
    - `downsample [percentage: float]`: Down-sample the workload
  - *Parameters*:
    - `vis [method <0, 1, 2>: int]`: Visualization methods. (0: Smoothed Histogram steps, 1: Histogram bar chart, 2: Histogram + KDE).
    - `bw [value: float]`: bw for KDE.
    - `bins [value: int]`: Number of bins for Histograms
  - *Switches*: 
    - `save [option <on, off>: str]`: Toggle automatic saving
    - `show [option <on, off>: str]`: Toggle automatic visualization display 

**Uniform high freq, skewed low freq**: 

`python ./main.py -a1 22 -a2 2.2 -n 10000 -d 50000 -g 100 && python ./vis_plan.py`

![Figure_1](/workload-generator/assets/Figure_1.png)

**Skewed high freq, uniform low freq**: 

`python ./main.py -a1 1.75 -a2 22 -n 10000 -d 50000 -g 10 && python ./vis_plan.py`

![Figure_5](/workload-generator/assets/Figure_4.png)

**Skewed both freq**: 

`python ./main.py -a1 1.5 -a2 1.1 -n 10000 -d 50000 -g 100 && python ./vis_plan.py`

![Figure_2](/workload-generator/assets/Figure_2.png)

**Skewed both freq, less groups**:

`python ./main.py -a1 1.5 -a2 1.5 -n 10000 -d 50000 -g 10 && python ./vis_plan.py`

![Figure_3](/workload-generator/assets/Figure_3.png)

**Uniform both freq**: 

`python ./main.py -a1 22 -a2 22 -n 10000 -d 50000 -g 10 && python ./vis_plan.py`

![Figure_4](/workload-generator/assets/Figure_0.png)


**AnalyticDB-V**:   

![Figure_5](/workload-generator/onormal_inormal_n500000_d2400_g3_a11.0_b10.2_a21.6_b20.4_s0.png)

**Twitter**:   

![Figure_6](/workload-generator/onormal_ipoisson_n50000_d1000_g10_a10.2_b10.7_a280.0_b20.7_s0.png)

**High bursty**:   

![Figure_7](/workload-generator/(high)ozipf_izipf_n50000_d100000_g1000_a11.15_b10.1_a21.05_b20.01_s0.png)

**Medium bursty**:   

![Figure_8](/workload-generator/(medium)ozipf_iuniform_n50000_d1000_g200_a12.2_b10.5_a21.001_b20.01_s0.png)

**Low bursty**:  

![Figure_9](/workload-generator/(low)oinverse_gaussian_inormal_n50000_d100000_g15_a11.15_b10.96_a25.0_b20.9_s0.png)
