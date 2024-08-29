import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras

# from model_capsule_want_fusion_signal import MFBasedModel

# from Integration.model_MF_integration import MFBasedModel

from OtherModel.DNN_based_capsule import DNNBasedModel
from OtherModel.GMF_based_capsule import GMFBasedModel

# from model_capsule import MFBasedModel
# from models import MFBasedModel

import numpy as np
from torchpq.clustering import KMeans

import faiss


class Run:
    def __init__(self, config):
        self.use_cuda = config["use_cuda"]
        self.lamd = config["lamd"]
        self.tau = config["tau"]
        self.K = config["K"]
        self.interest_num = config["interest_num"]
        self.prot_K = config["prot_K"]
        self.prot_alpha = config["prot_alpha"]
        ##note:wo args
        self.wo_att = config["wo_att"]
        self.wo_att_proj = config["wo_att_proj"]
        self.wo_mutli_inter = config["wo_mutli_inter"]
        self.wo_capsule = config["wo_capsule"]
        self.wo_disagree = config["wo_disagree"]

        ##note:new abaltion
    
        self.wo_adaptive=config["wo_adaptive"] 
        self.wo_prototype=config["wo_prototype"] 
        self.wo_tgt=config["wo_tgt"] 
        self.wo_interest=config["wo_interest"] 
        ####

        #### 


        self.charge = config["charge"]
        self.base_model = config["base_model"]
        self.root = config["root"]
        self.ratio = eval(str(config["ratio"]))
        self.task = config["task"]
        self.src = config["src_tgt_pairs"][self.task]["src"]
        self.tgt = config["src_tgt_pairs"][self.task]["tgt"]
        self.uid_all = config["src_tgt_pairs"][self.task]["uid"]
        self.iid_all = config["src_tgt_pairs"][self.task]["iid"]
        self.batchsize_src = config["src_tgt_pairs"][self.task]["batchsize_src"]
        self.batchsize_tgt = config["src_tgt_pairs"][self.task]["batchsize_tgt"]
        self.batchsize_meta = config["src_tgt_pairs"][self.task]["batchsize_meta"]
        self.batchsize_map = config["src_tgt_pairs"][self.task]["batchsize_map"]
        self.batchsize_test = config["src_tgt_pairs"][self.task]["batchsize_test"]
        self.batchsize_aug = self.batchsize_src

        self.epoch = config["epoch"]
        self.cross_epoch = config["cross_epoch"]
        self.emb_dim = config["emb_dim"]
        self.meta_dim = config["meta_dim"]
        self.num_fields = config["num_fields"]
        self.lr = config["lr"]
        self.wd = config["wd"]

        self.input_root = self.root + "ready/_" + str(int(self.ratio[0] * 10)) + "_" + str(int(self.ratio[1] * 10)) + "/tgt_" + self.tgt + "_src_" + self.src
        self.src_path = self.input_root + "/train_src.csv"
        self.tgt_path = self.input_root + "/train_tgt.csv"
        self.meta_path = self.input_root + "/train_meta.csv"
        self.test_path = self.input_root + "/test.csv"

        self.results = {"tgt_mae": 10, "tgt_rmse": 10, "aug_mae": 10, "aug_rmse": 10, "emcdr_mae": 10, "emcdr_rmse": 10, "ptupcdr_mae": 10, "ptupcdr_rmse": 10}

    def seq_extractor(self, x):  ##将交互从string=>int
        x = x.rstrip("]").lstrip("[").split(", ")
        for i in range(len(x)):  ##pos_log string-->int
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def run_kmeans(self, x, K):
        print("performing kmeans clustering")
        results = {"im2cluster": [], "centroids": [], "density": []}
        for seed, num_cluster in enumerate([K]):
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 100
            clus.nredo = 20
            # clus.niter = 50
            # clus.nredo = 10
            clus.seed = seed
            clus.max_points_per_centroid = 1000
            clus.min_points_per_centroid = 10
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = 0
            index = faiss.GpuIndexFlatL2(res, d, cfg)

            clus.train(x, index)

            D, I = index.search(x, 1)
            im2cluster = [int(n[0]) for n in I]

            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])
            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                    density[i] = d
            dmax = density.max()
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax
            density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
            density = (0.07) * density / density.mean()
            centroids = torch.Tensor(centroids).cuda()
            centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
            im2cluster = torch.LongTensor(im2cluster).cuda()
            density = torch.Tensor(density).cuda()
            results["centroids"].append(centroids)
            results["density"].append(density)
            results["im2cluster"].append(im2cluster)
            return results

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ["uid", "iid", "y"]
            x_col = ["uid", "iid"]
            y_col = ["y"]
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ["uid", "iid", "y", "pos_seq"]
            x_col = ["uid", "iid"]
            y_col = ["y"]
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding="post")
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)  ## sample_num *20
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)  ## sample_num *2
            X = torch.cat([id_fea, pos_seq], dim=1)  # (num_sample,22)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter

    def read_map_data(self):
        cols = ["uid", "iid", "y", "pos_seq", "tgt_pos_seq"]
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data["uid"].unique(), dtype=torch.long)  ## shape=(14425) 16738-test_user
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        cols_train = ["uid", "iid", "y"]
        x_col = ["uid", "iid"]
        y_col = ["y"]
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):
        print("========Reading data========")
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print("src {} iter / batchsize = {} ".format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print("tgt {} iter / batchsize = {} ".format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(
            self.meta_path,
            self.batchsize_meta,
            history=True,
        )
        print("meta {} iter / batchsize = {} ".format(len(data_meta), self.batchsize_meta))

        # data_map = self.read_map_data()
        # print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))
        data_map = None
        # data_aug = self.read_aug_data()
        # print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))
        data_aug = None
        data_test = self.read_log_data(
            self.test_path,
            self.batchsize_test,
            history=True,
        )
        print("test {} iter / batchsize = {} ".format(len(data_test), self.batchsize_test))

        return data_src, data_tgt, data_meta, data_map, data_aug, data_test

    def get_model(self):
        ##note: 消融引入
        if self.wo_adaptive:
            from model_capsule_want_fusion_wo_adaptive import MFBasedModel
        elif self.wo_prototype:
            from model_capsule_want_fusion_wo_prototype import MFBasedModel
        elif self.wo_tgt:
            from model_capsule_want_fusion_wo_tgt import MFBasedModel
        elif self.wo_interest:
            from model_capsule_want_fusion_wo_interest import MFBasedModel
        else:
            from model_capsule_want_fusion import MFBasedModel

        if self.base_model == "MF":
            model = MFBasedModel(
                self.uid_all,
                self.iid_all,
                self.num_fields,
                self.emb_dim,
                self.meta_dim,
                K=self.K,
                interest_num=self.interest_num,
                prot_alpha=self.prot_alpha,
                wo_att=self.wo_att,
                wo_att_proj=self.wo_att_proj,
                wo_mutli_inter=self.wo_mutli_inter,
                wo_capsule=self.wo_capsule,
            )
        elif self.base_model == "DNN":
            model = DNNBasedModel(
                self.uid_all,
                self.iid_all,
                self.num_fields,
                self.emb_dim,
                self.meta_dim,
                K=self.K,
                interest_num=self.interest_num,
                prot_alpha=self.prot_alpha,
                wo_att=self.wo_att,
                wo_att_proj=self.wo_att_proj,
                wo_mutli_inter=self.wo_mutli_inter,
                wo_capsule=self.wo_capsule,
            )
        elif self.base_model == "GMF":
            model = GMFBasedModel(
                self.uid_all,
                self.iid_all,
                self.num_fields,
                self.emb_dim,
                self.meta_dim,
                K=self.K,
                interest_num=self.interest_num,
                wo_att=self.wo_att,
                wo_att_proj=self.wo_att_proj,
                wo_mutli_inter=self.wo_mutli_inter,
                wo_capsule=self.wo_capsule,
            )
        else:
            raise ValueError("Unknown base model: " + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(
            [
                # {"params": model.meta_net.parameters()},
                # {"params": model.public_bridge.parameters()},
                # {"params": model.cate_bridge.parameters()},
                {"params": model.capsule_bridge.parameters()},
                {"params": model.capsule_mapping_cross_user_embedd_K.parameters()},
                {"params": model.capsule_mapping_cross_user_embedd_V.parameters()},
                {"params": model.capsule_mapping_cross_user_embedd_prot_K.parameters()},
                {"params": model.capsule_mapping_cross_user_embedd_prot_V.parameters()},
                {"params": model.dynamic_fusion.parameters()},
            ],
            lr=self.lr,
            weight_decay=self.wd,
        )
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_map = torch.optim.Adam(params=model.public_bridge.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map

    def eval_mae(self, model, data_loader, stage):
        print("Evaluating MAE:")
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    # ##
    # def cluster(self, embedd, K):
    #     km = KMeans(n_clusters=K)  # 先定K个聚类中心
    #     label = km.fit_predict((embedd).cpu().detach().numpy())
    #     return label  # 计算簇中心以及为簇分配序号
    #     # expenses = np.sum(km.cluster_centers_,axis=1)  #聚类中心点的数值加和

    def train(self, data_loader, model, criterion, optimizer, epoch, stage, K=512, mapping=False):
        print("Training Epoch {}:".format(epoch + 1))
        model.train()
        if mapping:
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):  # X:(256,2),一个用户id, item_id,y:评分
                src_emb, tgt_emb = model(X, stage, tau=self.tau)
                loss = criterion(src_emb, tgt_emb)
                model.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            if stage == "train_meta":
                ## 不使用这块
                if epoch == 0:

                    # print("聚类")
                    # uid_embedding_weights = model.src_model.uid_embedding.weight
                    # print("K=:", self.K)
                    # uid_cluser_labels = self.cluster(uid_embedding_weights, K=self.K)
                    # model.uid_cluser_labels = torch.from_numpy(uid_cluser_labels).to(device="cuda")

                    print("聚类_starting")
                    if self.base_model == "MF":
                        src_uid_embedding_weights = model.src_model.uid_embedding.weight.detach().contiguous()
                        tgt_iid_embedding_weights = model.tgt_model.iid_embedding.weight.detach().contiguous()

                    else:
                        src_uid_embedding_weights = model.src_model.embedding.uid_embedding.weight.detach().contiguous()
                        tgt_iid_embedding_weights = model.tgt_model.embedding.iid_embedding.weight.detach().contiguous()
                    # print(uid_embedding_weights.shape)
                    src_uid_embedding_weights = src_uid_embedding_weights.cpu().numpy()
                    src_uid_embedding_weights = np.ascontiguousarray(src_uid_embedding_weights)
                    # src_res = self.run_kmeans(src_uid_embedding_weights, self.prot_K)
                    # model.uid_cluser_labels = src_res["im2cluster"][0]
                    # model.uid_cluser_prototypes_embedding = src_res["centroids"][0]

                    # print(uid_embedding_weights.shape)
                    tgt_iid_embedding_weights = tgt_iid_embedding_weights.cpu().numpy()
                    tgt_iid_embedding_weights = np.ascontiguousarray(tgt_iid_embedding_weights)
                    print(tgt_iid_embedding_weights.shape)
                    tgt_res = self.run_kmeans(tgt_iid_embedding_weights, self.prot_K)
                    model.tgt_iid_cluser_labels = tgt_res["im2cluster"][0]
                    model.tgt_iid_cluser_prototypes_embedding = tgt_res["centroids"][0]

                    print("聚类_finished")
                ##
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):  # X:(256,2),一个用户id, item_id,y:评分

                    pred_orgin, pred_gen, cl_loss, l2_loss, disgreement_loss = model(X, stage, tau=self.tau)
                    loss1 = criterion(pred_orgin, y.squeeze().float())
                    # loss2 = criterion(pred_gen, y.squeeze().float())

                    loss = loss1 + disgreement_loss + l2_loss + 0.1 * cl_loss
                    # loss = loss1

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):  # X:(256,2),一个用户id, item_id,y:评分
                    pred = model(X, stage, tau=self.tau)
                    loss = criterion(pred, y.squeeze().float())
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + "_mae"]:
            self.results[phase + "_mae"] = mae
        if rmse < self.results[phase + "_rmse"]:
            self.results[phase + "_rmse"] = rmse

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        print("=========TgtOnly--只能训练目标域中的item  embeding========")
        for i in range(self.epoch):
            self.train(data_tgt, model, criterion, optimizer, i, stage="train_tgt")
            print("test_tgtOnly")
            mae, rmse = self.eval_mae(model, data_test, stage="test_tgt")
            self.update_results(mae, rmse, "tgt")
            print("MAE: {} RMSE: {}".format(mae, rmse))

    def CDR(self, model, data_src, data_map, data_meta, data_test, criterion, optimizer_src, optimizer_map, optimizer_meta, charge):
        if charge == False:
            # note:charge
            print("=====CDR Pretraining--训练源域中的user-embedding,item-embedding=====")
            for i in range(self.epoch):
                self.train(data_src, model, criterion, optimizer_src, i, stage="train_src")

        #

        # print('==========EMCDR==========')
        # for i in range(self.epoch):
        #     self.train(data_map, model, criterion, optimizer_map, i, stage='train_map', mapping=True)
        #     mae, rmse = self.eval_mae(model, data_test, stage='test_map')
        #     self.update_results(mae, rmse, 'emcdr')
        #     print('MAE: {} RMSE: {}'.format(mae, rmse))
        print("==========PTUPCDR--metanet进行特征跨域转移=========")
        for i in range(self.cross_epoch):  ## default:self.epoch
            self.train(data_meta, model, criterion, optimizer_meta, i, stage="train_meta")
            mae, rmse = self.eval_mae(model, data_test, stage="test_meta")
            self.update_results(mae, rmse, "ptupcdr")
            print("MAE: {} RMSE: {}".format(mae, rmse))

    def main(self):
     

        model = self.get_model()
        data_src, data_tgt, data_meta, data_map, data_aug, data_test = self.get_data()
        optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()
        if self.charge == False:
            # note:charge
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            #
        # self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
        self.CDR(model, data_src, data_map, data_meta, data_test, criterion, optimizer_src, optimizer_map, optimizer_meta, self.charge)
        print(f"task:{self.task},ratio:{self.ratio},lr:{self.lr},epoch:{self.epoch}")
        print(self.results)
