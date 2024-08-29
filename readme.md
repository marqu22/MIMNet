## Dataset handle

We utilized the Amazon Reviews 5-score dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.

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