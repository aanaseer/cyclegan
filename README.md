# CycleGAN

This is repository contains a re-implementation of the CycleGAN architecture found in [[1]](https://arxiv.org/abs/1703.10593). 
Code in this repository is submitted as part of an assignment for the Oxford Trinity Term 2022 
[Python in Scientific Computing](https://courses.maths.ox.ac.uk/course/view.php?id=252).

## Installation and Running
1. Download the zip file containing preconfigured directories and datasets from [here](https://drive.google.com/file/d/1OUuYDjaqEvZPP6Jrnc8IIpTjwyCY_zfE/view?usp=sharing). 
2. Extract the zip file (and all other zip files within) and place contents inside the root directory.
3. Set up a virtual environment and install the dependencies using the requirements.txt file by running
    ```
    pip install -r requirements.txt
    ```
4. Amend the configurations dictionary as required in the file train.py.   
5. Train a model by choosing a relevant GPU device with the following command
    ```
    CUDA_VISIBLE_DEVICES=[GPU device] python src/train.py
    ```
   To run without a GPU use,
    ```
    python train.py
    ```
6. Run evaluations by adjusting the configurations dictionary in evaluate.py according to the 
required settings and by running,
    ```
    python src/evaluate.py
    ```
7. To compute metrics specify the desired dataset in metrics.py file and then run,
    ```
    python src/metrics.py
    ```
   Currently metrics can only be computed on a CUDA enabled GPU.

## Pre-trained Models
Pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1zsF6Yqa-m-sAeeenH5wU4TfsmKwa5qk_?usp=sharing).
