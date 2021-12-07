# Towards All-Weather Autonomous Driving

<img src="imgs/overview.jpeg" width="800px"/>

### [Paper](https://arxiv.org/abs/) | [Pretrained Models](https://www.dropbox.com/sh/)

In this work, we propose deep learnin based MMSFM that is generalizable to diverse settings such as day, night, rain, fog, and snow. MMSFM is a geometry-Aware, multi-modal, modular, interpretable, and self-supervised ego-motion estimation system.

MMSFM involves the following architectures:

- Self-supervised monocular depth and ego-motion
- Self-supervised stereo depth and ego-motion
- Self-supervised lidar ego-motion
- Self-supervised radar ego-motion
- Self-supervised depth and ego-motion from camera and lidar
- Self-supervised depth and ego-motion from camera and radar

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

## Datasets
MMSFM currently supports the following datasets and the sensors. To easily make use of our preprocessing codes, keep the directory structure the same as the original datasets.  Please follow their licencing regulations for the datasets.
- [Oxford Robotcar](https://robotcar-dataset.robots.ox.ac.uk/) dataset 
  - Bumblebee XB3 stereo camera (main data source for the experiments in the paper)
  - Grasshopper2 monocular camera
  - SICK LD-MRS 3D LIDAR
  - NovAtel SPAN-CPT ALIGN inertial and GPS navigation system
- [Oxford Radar Robotcar](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/) dataset
  - Bumblebee XB3
  - Grasshopper2
  - NovAtel SPAN-CPT ALIGN inertial and GPS navigation system
  - Navtech CTS350-X radar
  - Velodyne HDL-32E lidar
- [RADIATE](http://pro.hw.ac.uk/radiate/) dataset.
  - ZED stereo camera
  - Velodyne HDL-32E lidar
  - Navtech CTS350-X radar
  - Advanced Navigation Spatial Dual GPS/IMU

To test our provided pretrained models, you can download the small-size sample sequences.

### Preprocessing
Although MMSFM supports on-the-fly processing of the input files, we recommend an offline processing of the input datasets using the provided scripts. The scripts will process the datasets and save the processed files to disk. The preprocessing includes colour-demosaicing of the Bayer images and rectification. We also optionally crop the bottom part of the Robotcar images occluded by the bonnet. You can use the following script for preprocessing:

For the Robotcar dataset:
```shell
cd preprocess/
python undistort_robotcar.py /dir/to/robotcardataset/
```

For the RADIATE dataset:
```shell
cd preprocess/
python rectify_radiate.py /dir/to/radiatedataset/
```

The scripts will create a sub-directory in the dataset folder named `stereo_undistorted` and save the processed files to the directory. You can notify the training and inference codes passing the `pretrained` flag. See the documentation of the codes for more details.

## Experiments
You can use the following commands to train and test the models.

### Training
After the download and the preprocessing, you can run the following commands. For ease of use, we created individual files to train/test the desired modalities.

Radar-only self-supervised ego-motion estimation:
```shell
cd preprocess/
python rectify_radiate.py /dir/to/radiatedataset/
```

