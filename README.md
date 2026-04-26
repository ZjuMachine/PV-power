# Rethinking the Use of Deep Learning Methods for Photovoltaic Power Forecasting

> Full code and datasets for the paper **“Rethinking the Use of Deep Learning Methods for Photovoltaic Power Forecasting”**

## News

- **2026-04-23**: We have uploaded the full dataset in: https://huggingface.co/datasets/yujiaA/AI-PVOD
- **2026-04-16**: This paper has been accepted by **Nature Communications**.
- **2025-12-03**: We have added a new dataset, called AI-PVOD, including the AI weather model forecast data in PV power forecasting. The readers can get it from: 

链接：https://pan.baidu.com/s/1CLRdp7BCNJGZp0qAfT_zuw 
提取码：1234 

- **2025-06-04**: We submitted it to **Nature Communications**.

## Authors

- **Yujia Zhang**
- **Yuzhou Zhang** *(Corresponding Author)*
- **Zhixiang Dai**
- **Rita Zhang**

## Affiliation

**NVIDIA Corporation, Beijing, China**

---

## Overview

This repository provides the **full codebase** and **datasets** used in the paper:

**“Rethinking the Use of Deep Learning Methods for Photovoltaic Power Forecasting”**

The repository includes:

- PV forecasting model implementation
- baselines in original paper
- Datasets used in the paper
- Numerical Weather Prediction (NWP) forecast data
- Satellite-observed irradiance data
- Additional AI weather model forecast data via **AI-PVOD**

---

## Abstract
Abstract: Accurate photovoltaic (PV) power forecasting remains critical for grid stability but is challenged by weather uncertainties and the inability of traditional methods to effectively integrate historical observations with forward-looking meteorological information. We revisit existing deep learning (DL) sequential model architectural choices, and demonstrate the critical importance of full encoder-decoder architectures and channel dependence modeling when both forward-looking weather forecasts and historical data are available. Based on this insight, we propose Cross-Unet, a Transformer-based architecture featuring multi-scale temporal encoding, correlation-aware channel attention, and hierarchical cross-attention decoding to effectively fuse historical generation data with weather forecasts. We evaluate Cross-Unet on open-source datasets from four utility-scale plants in northern China and one aggregated plant in central Australia, using three types of forward-looking inputs: numerical weather prediction, satellite-derived irradiance, and AI-based weather model forecasts. Across the majority of evaluated configurations spanning 5 PV stations, 5 forecasting horizons (4 hours to 7 days), and 3 forecast sources, Cross-Unet outperforms ten deep learning baselines and traditional operational benchmarks. By integrating advanced forecasting systems, such as modern AI weather models, into an end-to-end forecasting pipeline, Cross-Unet enables operational 15-minute-resolution predictions over 4-hour to 7-day horizons, supporting grid scheduling and energy trading.


## Quick Start

0. Environment preparation:  (within 15 min when the network is unobstructed)
Execute the these commands in the same directory with this readme file:
```
conda create -n [your env name] python=3.10
conda activate [your env name]
pip install -r requirements.txt
```
1. Data preparation
The data used in this paper is downloaded from: DKA Solar Centre (https://dkasolarcentre.com.au/)
and Science Data Bank (https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62). You can also donloaded from https://huggingface.co/datasets/yujiaA/AI-PVOD (pre-processed full dataset)

As for the AI weather model forecast dataset, Please download the folder from https://github.com/ZjuMachine/PV-power/tree/main. Then, put the folder "AIweathermodel" into the folder: "dataset/"

2. Train and test:
For a simple demo, run this command in the same directory with this readme file. This command will use KDASC.csv to train cross-unet model for forecasting window of 1 day:
```
python use_cross_unet.py
```

This command trains cross_unet model with the AI weather model forecast:

```
python use_cross_unet.py --deployment --station_name='S-1'
```

For reproducing all the results in this paper, we provide python scripts for every model, and you can use command line parameters to choose dataset and weather to use solar radiation data from satellite Himawari-8 or from NWP. For example, this command trains patchtst model with S-1 dataset and solar radiation data from the satellite:
```
python use_patchtst.py --station_name='S-1' --use_satell
```
And this command trains cyclenet model with S-2 dataset and solar radiation data from NWP
```
python use_cyclenet.py --station_name='S-2'
```

The hyperparameters for a specific model are taken from its original paper. Available models include : cross_unet, crossformer, cyclenet, paifilter, patchmlp, patchtst, timemixer, timesnet, transformer

3. The test results (including visualization results and metrics) will be saved to 'results' and 'test_results'. The weights of the cross-unet will be saved to 'checkpoints'

Note: This code can  run on an NVIDIA A100  GPU (with a memory size of 40GB or 80GB), and most of the running configurations can finish in 20-40 minutes for one model training


