# MIMNet: Multi-Interest Meta Network with Multi-Granularity
 Target-Guided Attention for Cross-domain Recommendation

## Dataset

The Amazon datasets we used: 
1. CDs and Vinyl: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
2. Movies and TV: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz  
3. Books: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz

Put the data files in `./data/raw`.

Data process via:
```python
python entry.py --process_data_mid 1 --process_data_ready 1
```
```

└── data
    ├── mid                 # Mid data
    │   ├── Books.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Movies_and_TV.csv
    ├── raw                 # Raw data
    │   ├── reviews_Books_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Movies_and_TV_5.json.gz
    └── ready               # Ready to use
        ├── _2_8
        ├── _5_5
        └── _8_2

```
For the sake of simplicity, we provide the processed data, which can be used directly from link[https://pan.baidu.com/s/146zkWoYBjTNiOAVoHeu24w?pwd=ABCD].

## env
```
python==3.8.0
torch
faiss-gpu
pandas
numpy
tensorflow==2.7.0
```
## model run 
```
python entry_self_capsule_want_ablation.py --task 1 --ratio [0.8,0.2] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 1 --ratio [0.5,0.5] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 1 --ratio [0.2,0.8] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 2 --ratio [0.8,0.2] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 2 --ratio [0.5,0.5] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 2 --ratio [0.2,0.8] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 3 --ratio [0.8,0.2] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 3 --ratio [0.5,0.5] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
python entry_self_capsule_want_ablation.py --task 3 --ratio [0.2,0.8] --epoch 10 --lr 0.01 --interest_num 7 --prot_K 100 --base_model MF --seed 2020  
```
## cite
[CMF] Ajit P Singh and Geoffrey J Gordon. 2008. Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 650–658.


[EMCDR] Tong Man, Huawei Shen, Xiaolong Jin, and Xueqi Cheng. 2017. Cross-domain recommendation: An embedding and mapping approach.. In IJCAI, Vol. 17. 2464–2470.


[PTUPCDR] Yongchun Zhu, Zhenwei Tang, Yudan Liu, Fuzhen Zhuang, Ruobing Xie, Xu Zhang, Leyu Lin, and Qing He. 2022. Personalized transfer of user preferences for cross-domain recommendation. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 1507–1515


[SSCDR] SeongKu Kang, Junyoung Hwang, Dongha Lee, and Hwanjo Yu. 2019. Semi-supervised learning for cross-domain recommendation to cold-start users. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1563–1572.


[CVPM] Zhao, Chuang, et al. "Cross-domain Transfer of Valence Preferences via a Meta-optimization Approach." arXiv preprint arXiv:2406.16494 (2024).


More hyper-parameter settings can be made in `./code/config_final.json`.
