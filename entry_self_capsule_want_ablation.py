from ast import arg
import os
import torch
import numpy as np
import random
import argparse
import json
from preprocessing_self import DataPreprocessingMid, DataPreprocessingReady
from run_self_capsule_want_ablation import Run


def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_data_mid", default=0)
    parser.add_argument("--process_data_ready", default=0)
    parser.add_argument("--task", default="1")
    parser.add_argument("--base_model", default="MF")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--ratio", default="[0.8, 0.2]")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--cross_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lamd", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--interest_num", type=int, default=3)
    parser.add_argument("--prot_K", type=int, default=50)
    parser.add_argument("--prot_alpha", type=float, default=0.5)
    parser.add_argument("--charge", type=bool, default=False)
    ## note:消融选项
    parser.add_argument("--wo_att", type=int, default=0)
    parser.add_argument("--wo_att_proj", type=int, default=0)
    parser.add_argument("--wo_mutli_inter", type=int, default=0)
    parser.add_argument("--wo_capsule", type=int, default=0)
    parser.add_argument("--wo_disagree", type=int, default=0)

    ## note:消融选项
    parser.add_argument("--wo_adaptive", action="store_true", default=False)
    parser.add_argument("--wo_prototype", action="store_true", default=False)
    parser.add_argument("--wo_tgt", action="store_true", default=False)
    parser.add_argument("--wo_interest", action="store_true", default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, "r") as f:
        config = json.load(f)
        config["base_model"] = args.base_model
        config["task"] = args.task
        config["ratio"] = args.ratio
        config["epoch"] = args.epoch
        config["lr"] = args.lr
        config["lamd"] = args.lamd
        config["tau"] = args.tau
        config["K"] = args.K
        config["interest_num"] = args.interest_num
        config["charge"] = args.charge

        config["prot_K"] = args.prot_K
        config["prot_alpha"] = args.prot_alpha
        config["cross_epoch"] = args.cross_epoch
        ##ABATION
        config["wo_att"] = args.wo_att
        config["wo_att_proj"] = args.wo_att_proj
        config["wo_mutli_inter"] = args.wo_mutli_inter
        config["wo_capsule"] = args.wo_capsule
        config["wo_disagree"] = args.wo_disagree
        ##new ABATION
        config["wo_adaptive"] = args.wo_adaptive
        config["wo_prototype"] = args.wo_prototype
        config["wo_tgt"] = args.wo_tgt
        config["wo_interest"] = args.wo_interest

    return args, config


if __name__ == "__main__":
    config_path = "config_final.json"
    args, config = prepare(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.process_data_mid:
        for dealing in ["Books", "CDs_and_Vinyl"]:
            DataPreprocessingMid(config["root"], dealing).main()
    if args.process_data_ready:
        for ratio in [[0.8, 0.2]]:
            for task in ["3"]:
                DataPreprocessingReady(config["root"], config["src_tgt_pairs"], task, ratio).main()
    print(
        "task:{}; model:{}; ratio:{}; epoch:{};cross_epoch_5:{} lr:{}; gpu:{}; seed:{};lamd:{};tau:{};interest_num:{};charge:{};  prot_K:{};prot_alpha:{}; wo_att:{};wo_att_proj:{};wo_mutli_inter:{};wo_capsule:{};wo_disagree:{} ;wo_adaptive:{};wo_prototype:{};wo_tgt:{};wo_interest:{}".format(
            args.task,
            args.base_model,
            args.ratio,
            args.epoch,
            args.cross_epoch,
            args.lr,
            args.gpu,
            args.seed,
            args.lamd,
            args.tau,
            args.interest_num,
            args.charge,
            args.prot_K,
            args.prot_alpha,
            args.wo_att,
            args.wo_att_proj,
            args.wo_mutli_inter,
            args.wo_capsule,
            args.wo_disagree,
            args.wo_adaptive,
            args.wo_prototype,
            args.wo_tgt,
            args.wo_interest,
        )
    )

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main()
