# Generalized Graph Signal Sampling by Difference-of-Convex Optimization

This repository provides the official MATLAB demonstration code for the algorithms proposed in our paper: **"Generalized Graph Signal Sampling by Difference-of-Convex Optimization"** (Available on [arXiv:2306.14634](https://arxiv.org/abs/2306.14634)).

Our framework designs sampling operators for generalized graph signals under arbitrary graph signal priors via difference-of-convex (DC) optimization and executes signal recovery.

---

## Prerequisites

Before running the scripts, ensure you have the following installed:
* **MATLAB** (R2022b or later recommended)
* **GSPBox** (Graph Signal Processing Toolbox)  
  Please install the toolbox from the official website: [https://epfl-lts2.github.io/gspbox-html/](https://epfl-lts2.github.io/gspbox-html/)

---

## File Structure

The repository consists of the following core files:
* `demo.m`: The main integrated demo script. It handles graph and graph signal generation, user-defined parameter settings, and results visualization.
* `generalized_gs_sampling_by_dc.m`: The core optimization pipeline implementing the proposed DC loop and graph signal recovery.

---

## How to Run

1. Clone or download this repository to your local machine.
2. Add the directory to your MATLAB path.
3. Open `demo.m` in MATLAB.
4. Select your desired prior profile (`signal_type = 'BL'`, `'PWL'`, or `'SGS'`) and sampling design (`design_choices = [1]`, `[2]`, or `[3]`) in Section 1.
5. Run the script. The quantitative metrics (MSE, execution time) will be printed in the Command Window, and a figure showing the comparison between the original and recovered graph signals will be displayed.

---

## Citation

If you use this code or find our method helpful in your research, please cite our arXiv preprint:

```bibtex
@misc{yamashita2026generalized,
  title={Generalized Graph Signal Sampling by Difference-of-Convex Optimization}, 
  author={Keitaro Yamashita and Kazuki Naganuma and Shunsuke Ono},
  year={2026},
  eprint={2306.14634},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2306.14634},
  howpublished = {\textit{arXiv}},
}
