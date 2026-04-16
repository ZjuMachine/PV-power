This is the Full code and dataset (including the NWP forecast and satllite-observed irradiance) of the paper "Rethinking the Use of Deep Learning Methods for Photovoltaic Power Forecasting"

Authors: Yujia Zhang, Yuzhou Zhang (corresponding author), Zhixiang Dai, Rita Zhang

Affiliation: NVIDIA Corporation, Beijing, China 

Abstract: Accurate photovoltaic (PV) power forecasting remains critical for grid stability but is challenged by weather uncertainties and the inability of traditional methods to effectively integrate historical observations with forward-looking meteorological information. We revisit existing deep learning (DL) sequential model architectural choices, and demonstrate the critical importance of full encoder-decoder architectures and channel dependence modeling when both forward-looking weather forecasts and historical data are available. Based on this insight, we propose Cross-Unet, a Transformer-based architecture featuring multi-scale temporal encoding, correlation-aware channel attention, and hierarchical cross-attention decoding to effectively fuse historical generation data with weather forecasts. We evaluate Cross-Unet on open-source datasets from four utility-scale plants in northern China and one aggregated plant in central Australia, using three types of forward-looking inputs: numerical weather prediction, satellite-derived irradiance, and AI-based weather model forecasts. Across the majority of evaluated configurations spanning 5 PV stations, 5 forecasting horizons (4 hours to 7 days), and 3 forecast sources, Cross-Unet outperforms ten deep learning baselines and traditional operational benchmarks. By integrating advanced forecasting systems, such as modern AI weather models, into an end-to-end forecasting pipeline, Cross-Unet enables operational 15-minute-resolution predictions over 4-hour to 7-day horizons, supporting grid scheduling and energy trading.

2026/04/16: This paper has been accepted in Nature Communications.

2025/12/03: We have added a new dataset, called AI-PVOD, including the AI weather model forecast data in PV power forecasting. The readers can get it from: 

链接：https://pan.baidu.com/s/1CLRdp7BCNJGZp0qAfT_zuw 
提取码：1234 

