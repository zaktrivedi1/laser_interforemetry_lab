## About

- Programme to analyse data read from a photodiode, which was used as the output to a laser interferometer. 
- Example of data analysis carried out in a few weeks of my lab work - similar was done throughout the 3 year course. 
- Most of the data files were omitted from this repository for brevity but one of each was kept so they can be tested here.

## Features

- Estimates the dominant modulation frequency using **Lomb–Scargle** and **FFT**, compute the **period** and **uncertainty**, derive **fringe visibility** (with error propagation), and—when not a pure laser—fit a **Gaussian visibility envelope** to estimate **coherence length** and **spectral linewidth**.
-   `plotting_interferograms.py` — analysis of one source (e.g., white LED)
-   `two_sources.py` — laser channel used to infer mirror positions; LED channel analysed at the same positions
