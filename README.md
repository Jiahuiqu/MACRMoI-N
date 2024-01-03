# Joint Contextual Representation Model-Informed Interpretable Network With Dictionary Aligning for Hyperspectral and LiDAR Classification

## This example implements the paper in review [Joint Contextual Representation Model-Informed Interpretable Network With Dictionary Aligning for Hyperspectral and LiDAR Classification]


## Prerequisites
- Python == 3.6

#### run command below to install other packages
```
pip install requirements
```

## Usage

### Data set links

Houston dataset were introduced for the 2013 IEEE GRSS Data Fusion contest. Data set links comes from
```
http://www.grss-ieee.org/community/technical-committees/data-fusion/2013-ieee-grss-data-fusion-contest/
```

### Dataset utilization

In
```
dataset.py
```

### Demo

Run demo in 
```
python demo.py 
```


## Results
All the results are cited from original paper. More details can be found in the paper.

| dataset  	 | OA | AA      |Kappa
|---------- |-------  |--------|--------
| Houston  | 90.09%| 91.50%|89.29%
| Trento    | 97.48%| 96.29% |96.65%

## Citation
@ARTICLE{10105921,
  author={Dong, Wenqian and Yang, Teng and Qu, Jiahui and Zhang, Tian and Xiao, Song and Li, Yunsong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Joint Contextual Representation Model-Informed Interpretable Network With Dictionary Aligning for Hyperspectral and LiDAR Classification}, 
  year={2023},
  volume={33},
  number={11},
  pages={6804-6818},
  doi={10.1109/TCSVT.2023.3268757}}
