# Towards All-Weather Autonomous Driving

<img src="imgs/overview.jpeg" width="800px"/>

### [Paper](https://arxiv.org/abs/) | [Pretrained Models](https://www.dropbox.com/sh/)

In this work, we propose deep learnin based MMSFM that is generalizable to diverse settings such as day, night, rain, fog, and snow. MMSFM is a geometry-Aware, multi-modal, modular, interpretable, and self-supervised ego-motion estimation system.

Overview of the MMSFM conceptual framework and architecture:
<br>
<img src='imgs/FIG-01.png' align="right" width=960>
<br>

## Example Results

### Generalization to adverse conditions and depth prediction performance.
<img src="imgs/FIG-02.png" width="800px"/>

### Generalization to adverse conditions and depth prediction performance.
<img src="imgs/FIG-04.png" width="800px"/>


## Computational Hardware and Software
The experiments are conducted on an Ubuntu 20.04 LTS computer with Intel Xeon CPUs and NVIDIA RTX 3090 GPUs accelerated by CuDNN, using Python 3.8.

### Getting started

- Clone this repo:
```bash
git clone https://github.com/yasinalm/mmsfm
cd mmsfm
```

### Prerequisites

<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts. -->
We recommend to create a virtual environment with Python 3.8 `conda create -n mmsfm python=3.8 anaconda`.
Using a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=1.8.2 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tensorboardx=2.4
pip install --user colour-demosaicing # needed to process datasets
```

You can install the remaining dependencies as follow:
```shell
conda install --file requirements.txt
```

