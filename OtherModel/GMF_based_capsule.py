import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.contrastiveLoss import ContrastiveLoss
from utils.CapsuleLayer import CapsuleLayer
import math


class LookupEmbedding(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))  ##查出对应用户的embedding
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))  ##查出对应item的embedding
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb


# note: 类别桥(聚类桥)定义
class Cate_bridge(torch.nn.Module):
    def __init__(self, K, emb_dim):
        super().__init__()
        self.cate_embedding = torch.nn.Embedding(K, emb_dim * emb_dim + emb_dim)
        # self.cate_project = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim * emb_dim))  ##eq 2

    def forward(self, x_cate):
        cate_embedd = self.cate_embedding(x_cate)
        # cate_bridge_mapping = self.cate_project(cate_embedd)
        return cate_embedd


# note: 个性化桥定义
class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim, traing_mode=True, aim_seq_inter=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1, False))  ##eq 2
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(), torch.nn.Linear(meta_dim, emb_dim * emb_dim))  ## eq3
        self.tgt_item_weights = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1, False))  ##eq 2

        self.tgt_item_aggr = torch.nn.Softmax(dim=1)
        self.project_tgt_aim2rsc_aim = torch.nn.Linear(emb_dim, emb_dim)
        self.Wq = torch.nn.Linear(emb_dim, emb_dim)

        self.Wk = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = torch.nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, rsc_inter_seq_emb, tgt_inter_seq_emb, seq_index, aim_domin_seq_inter=None, traing_mode=True):  ##tgt_目标域交互信息,seq_index源域交互信息
        if traing_mode == True:
            #
            mask = (seq_index == 0).float()  ## 0表示不mask
            # event_K = self.event_K(rsc_inter_seq_emb[:, :1])  ## event_k表示item对转移共享大小 约等于注意力
            event_K = self.event_K(rsc_inter_seq_emb)  ## event_k表示item对转移共享大小 约等于注意力
            t = event_K - torch.unsqueeze(mask, 2) * 1e8  # mask 空白
            att = self.event_softmax(t)  ## 计算Pui
            his_fea = torch.sum(att * rsc_inter_seq_emb[:, :1], 1)
            orgin_output = self.decoder(his_fea)  ## output ==Wui

            # 目标域引导的对比构造-目标域交互内容的平均,来判断源域中的哪些交互是重要的,并构造出一个目标域指导的mapping
            # tgt_item_mask = mask[:, 20:] # shape = 128*20
            # tgt_item_mask = tgt_item_mask.unsqueeze(2).expand(tgt_inter_seq_emb.size())# shape = 128*20*10
            # useful_emb = tgt_inter_seq_emb.masked_fill(tgt_item_mask == 1, 0) ##填充mask部分的embedding,防止影响目标域embedding的平均

            tgt_intent = tgt_inter_seq_emb.mean(dim=1)
            Q = tgt_intent  # shape=128*10
            Q = Q.unsqueeze(2)  # shape=128*10*1
            K = rsc_inter_seq_emb  # shape=128*20*10
            V = rsc_inter_seq_emb  # shape=128*20*10
            atten = torch.bmm(K, Q) / math.sqrt(self.emb_dim)
            atten1 = atten - torch.unsqueeze(mask[:, :20], 2) * 1e8  # 填充被mask的attention部分
            atten2 = self.event_softmax(atten1)
            pt_his_fea = torch.sum(atten2 * V, 1)
            gen_output = self.decoder(pt_his_fea)
            #
            #
            """
            tgt_intent = tgt_inter_seq_emb.mean(dim=1)
            
            Q = self.Wq(tgt_intent)  ##128*10
            Q = Q.unsqueeze(1)
            atten=torch.bmm(Q,rsc_inter_seq_emb.transpose(-2,-1))/math.sqrt(self.emb_dim)
            atten_masked = atten.transpose(1,2) - torch.unsqueeze(mask[:, :20], 2) * 1e8  # mask 空白
            atten = torch.nn.Softmax(dim=-1)(atten_masked)
            pt_his_fea =  torch.sum(atten * rsc_inter_seq_emb, 1)
            gen_output = self.decoder(pt_his_fea)
            #
            """

            # ## gambel-softmax
            # tgt_intent = tgt_inter_seq_emb.mean(dim=1)
            # Q = self.Wq(tgt_intent)  ##128*10
            # Q = Q.unsqueeze(1)
            # atten = torch.bmm(Q, rsc_inter_seq_emb.transpose(-2, -1)) / math.sqrt(self.emb_dim)
            # atten_masked = atten.transpose(1, 2) - torch.unsqueeze(mask[:, :20], 2) * 1e8  # mask 空白
            # gambel_atten = F.gumbel_softmax(atten.squeeze(1), tau=1, hard=False, dim=-1)
            # gambel_sample = torch.where(gambel_atten > (1 / 20), 1, 0).unsqueeze(dim=-1)

            # atten = torch.nn.Softmax(dim=-2)(atten_masked * gambel_sample)
            # pt_his_fea = torch.sum(atten * rsc_inter_seq_emb, 1)
            # gen_output = self.decoder(pt_his_fea)
            # return orgin_output.squeeze(1), gen_output.squeeze(1)

            # drop out method
            # dropout_rate = 0.1
            # drop_mask = torch.bernoulli(torch.where(mask == 1, 1.0, dropout_rate)).to(torch.int)  ## 0表示不mask
            # event_K = self.event_K(rsc_inter_seq_emb)  ## event_k表示item对转移共享大小
            # t = event_K - torch.unsqueeze(drop_mask[:, :20], 2) * 1e8  # mask 空白
            # att = self.event_softmax(t)  ## 计算Pui

            # pt_his_fea = torch.sum(att * rsc_inter_seq_emb, 1)
            # gen_output = self.decoder(pt_his_fea)

            return orgin_output.squeeze(1), gen_output.squeeze(1)

        else:  ## 推理阶段

            # mask = (seq_index == 0).float()  ## 0表示不mask
            # event_K = self.event_K(rsc_inter_seq_emb)  ## event_k表示item对转移共享大小
            # t = event_K - torch.unsqueeze(mask, 2) * 1e8  # mask 空白
            # att = self.event_softmax(t)  ## 计算Pui
            # his_fea = torch.sum(att * rsc_inter_seq_emb, 1)
            # output = self.decoder(his_fea)  ## output ==Wui

            # return output.squeeze(1)
            ## 测试少量interaction下的效果
            mask = (seq_index == 0).float()  ## 0表示不mask
            event_K = self.event_K(rsc_inter_seq_emb[:, :1])  ## event_k表示item对转移共享大小 约等于注意力
            t = event_K - torch.unsqueeze(mask[:, :1], 2) * 1e8  # mask 空白
            att = self.event_softmax(t)  ## 计算Pui
            his_fea = torch.sum(att * rsc_inter_seq_emb[:, :1], 1)
            orgin_output = self.decoder(his_fea)  ## output ==Wui
            return orgin_output.squeeze(1)


# note: 胶囊桥定义
class CapsuleBridge(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim, interest_num, traing_mode=True, aim_seq_inter=None, wo_capsule=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.wo_capsule = wo_capsule
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1, False))  ##eq 2
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(), torch.nn.Linear(meta_dim, emb_dim * emb_dim))  ## eq3
        self.decoder_simple = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * emb_dim))  ## eq3
        self.wo_decoder_1 = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * emb_dim))
        self.wo_decoder_2 = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * emb_dim))
        self.wo_decoder_3 = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * emb_dim))

        self.tgt_item_weights = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1, False))  ##eq 2

        self.tgt_item_aggr = torch.nn.Softmax(dim=1)
        self.project_tgt_aim2rsc_aim = torch.nn.Linear(emb_dim, emb_dim)
        self.Wq = torch.nn.Linear(emb_dim, emb_dim)

        self.Wk = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        # NOte: capsule:
        self.capsuleLayer = CapsuleLayer(10, 10, k_max=interest_num)

    def disgreement_loss(self, t1, t2):
        """
        * @description: 返回disagreement_loss
        * @param  self :
        * @param  t1 : 第一个batch的内容, (batch,interest,dim) at:(128,3,10)
        * @param  t2 : 第二个batch的内容, (batch,interest,dim) at:(128,3,10)
        * @return disagreement_loss
        """
        simil = torch.bmm(t1, t2.transpose(-1, -2))
        t1_L2 = torch.norm(t1, p=2, dim=-1, keepdim=True)
        t2_L2 = torch.norm(t2, p=2, dim=-1, keepdim=True)
        div = t1_L2 * t2_L2
        loss = (simil / div).mean(dim=-1).mean(dim=-1).mean(dim=-1)
        return loss

    def forward(self, rsc_inter_seq_emb, tgt_inter_seq_emb, seq_index, aim_domin_seq_inter=None, traing_mode=True):  ##tgt_目标域交互信息,seq_index源域交互信息
        if traing_mode == True:
            #
            mask = (seq_index[:, :20] == 0).float()  ## 0表示不mask
            seq_len = torch.where(mask == 1, 0, 1).sum(dim=1)
            seq_len = seq_len.unsqueeze(1)
            multi_interset = self.capsuleLayer(rsc_inter_seq_emb, seq_len)
            disagree_loss = self.disgreement_loss(multi_interset, multi_interset)
            # note: behavior_embs: [N, L, D] seq_len: [N, 1]
            # note: return : B * max_interest* dim
            # orgin_output = self.decoder_simple(multi_interset)
            orgin_output = self.decoder(multi_interset)

            ## important: 去掉胶囊
            if self.wo_capsule == 1:
                orgin_shape = orgin_output.shape
                mask = (seq_index == 0).float()  ## 0表示不mask
                event_K = self.event_K(rsc_inter_seq_emb)  ## event_k表示item对转移共享大小 约等于注意力
                t = event_K - torch.unsqueeze(mask, 2) * 1e8  # mask 空白
                att = self.event_softmax(t)  ## 计算Pui
                his_fea = torch.sum(att * rsc_inter_seq_emb, 1)
                wo_src_att = 1
                if wo_src_att == 1:
                    his_fea = torch.mean(rsc_inter_seq_emb, 1)

                orgin_output_1 = self.wo_decoder_1(his_fea.unsqueeze(dim=1))
                orgin_output_2 = self.wo_decoder_2(his_fea.unsqueeze(dim=1))
                orgin_output_3 = self.wo_decoder_3(his_fea.unsqueeze(dim=1))
                orgin_output_list = [orgin_output_1, orgin_output_1, orgin_output_1]
                orgin_output = torch.concat(orgin_output_list, dim=1)
                # orgin_output = orgin_output.unsqueeze(dim=1)
            ## important: 去掉胶囊

            # orgin_output = orgin_output.unsqueeze(dim=1)
            # orgin_output = orgin_output.expand(orgin_shape)  ## output ==Wui
            # 目标域引导的对比构造-目标域交互内容的平均,来判断源域中的哪些交互是重要的,并构造出一个目标域指导的mapping
            # tgt_item_mask = mask[:, 20:] # shape = 128*20
            # tgt_item_mask = tgt_item_mask.unsqueeze(2).expand(tgt_inter_seq_emb.size())# shape = 128*20*10
            # useful_emb = tgt_inter_seq_emb.masked_fill(tgt_item_mask == 1, 0) ##填充mask部分的embedding,防止影响目标域embedding的平均
            ## note:对比学习生成的embedding
            # tgt_intent = tgt_inter_seq_emb.mean(dim=1)
            # Q = tgt_intent  # shape=128*10
            # Q = Q.unsqueeze(2)  # shape=128*10*1
            # K = rsc_inter_seq_emb  # shape=128*20*10
            # V = rsc_inter_seq_emb  # shape=128*20*10
            # atten = torch.bmm(K, Q) / math.sqrt(self.emb_dim)
            # atten1 = atten - torch.unsqueeze(mask[:, :20], 2) * 1e8  # 填充被mask的attention部分
            # atten2 = self.event_softmax(atten1)
            # pt_his_fea = torch.sum(atten2 * V, 1)
            # gen_output = self.decoder(pt_his_fea)
            #
            #
            """
            tgt_intent = tgt_inter_seq_emb.mean(dim=1)
            
            Q = self.Wq(tgt_intent)  ##128*10
            Q = Q.unsqueeze(1)
            atten=torch.bmm(Q,rsc_inter_seq_emb.transpose(-2,-1))/math.sqrt(self.emb_dim)
            atten_masked = atten.transpose(1,2) - torch.unsqueeze(mask[:, :20], 2) * 1e8  # mask 空白
            atten = torch.nn.Softmax(dim=-1)(atten_masked)
            pt_his_fea =  torch.sum(atten * rsc_inter_seq_emb, 1)
            gen_output = self.decoder(pt_his_fea)
            #
            """

            # ## gambel-softmax
            # tgt_intent = tgt_inter_seq_emb.mean(dim=1)
            # Q = self.Wq(tgt_intent)  ##128*10
            # Q = Q.unsqueeze(1)
            # atten = torch.bmm(Q, rsc_inter_seq_emb.transpose(-2, -1)) / math.sqrt(self.emb_dim)
            # atten_masked = atten.transpose(1, 2) - torch.unsqueeze(mask[:, :20], 2) * 1e8  # mask 空白
            # gambel_atten = F.gumbel_softmax(atten.squeeze(1), tau=1, hard=False, dim=-1)
            # gambel_sample = torch.where(gambel_atten > (1 / 20), 1, 0).unsqueeze(dim=-1)

            # atten = torch.nn.Softmax(dim=-2)(atten_masked * gambel_sample)
            # pt_his_fea = torch.sum(atten * rsc_inter_seq_emb, 1)
            # gen_output = self.decoder(pt_his_fea)
            # return orgin_output.squeeze(1), gen_output.squeeze(1)

            # drop out method
            # dropout_rate = 0.1
            # drop_mask = torch.bernoulli(torch.where(mask == 1, 1.0, dropout_rate)).to(torch.int)  ## 0表示不mask
            # event_K = self.event_K(rsc_inter_seq_emb)  ## event_k表示item对转移共享大小
            # t = event_K - torch.unsqueeze(drop_mask[:, :20], 2) * 1e8  # mask 空白
            # att = self.event_softmax(t)  ## 计算Pui

            # pt_his_fea = torch.sum(att * rsc_inter_seq_emb, 1)
            # gen_output = self.decoder(pt_his_fea)

            return orgin_output.squeeze(1), orgin_output.squeeze(1), disagree_loss

        else:  ## 推理阶段
            mask = (seq_index[:, :20] == 0).float()  ## 0表示不mask
            seq_len = torch.where(mask == 1, 0, 1).sum(dim=1)
            seq_len = seq_len.unsqueeze(1)
            multi_interset = self.capsuleLayer(rsc_inter_seq_emb, seq_len)
            # note: behavior_embs: [N, L, D] seq_len: [N, 1]
            # note: return : B * max_interest* dim
            # orgin_output = self.decoder(multi_interset)
            orgin_output = self.decoder(multi_interset)
            # orgin_output=multi_interset
            ## important: 去掉胶囊
            if self.wo_capsule == 1:
                orgin_shape = orgin_output.shape
                mask = (seq_index == 0).float()  ## 0表示不mask
                event_K = self.event_K(rsc_inter_seq_emb)  ## event_k表示item对转移共享大小 约等于注意力
                t = event_K - torch.unsqueeze(mask, 2) * 1e8  # mask 空白
                att = self.event_softmax(t)  ## 计算Pui
                his_fea = torch.sum(att * rsc_inter_seq_emb, 1)
                wo_src_att = 1
                if wo_src_att == 1:
                    his_fea = torch.mean(rsc_inter_seq_emb, 1)

                orgin_output_1 = self.wo_decoder_1(his_fea.unsqueeze(dim=1))
                orgin_output_2 = self.wo_decoder_2(his_fea.unsqueeze(dim=1))
                orgin_output_3 = self.wo_decoder_3(his_fea.unsqueeze(dim=1))
                orgin_output_list = [orgin_output_1, orgin_output_1, orgin_output_1]
                orgin_output = torch.concat(orgin_output_list, dim=1)

            return orgin_output.squeeze(1)


class GMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0, :] * emb[:, 1, :]
        x = self.linear(x)
        return x.squeeze(1)


class GMFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0, K=512, interest_num=3, prot_alpha=0.5, wo_att=0, wo_capsule=0, wo_mutli_inter=0, wo_att_proj=0):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.interest_num = interest_num
        self.K = K

        self.wo_att = wo_att
        self.wo_capsule = wo_capsule

        self.src_model = GMFBase(uid_all, iid_all, emb_dim)
        self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        self.aug_model = GMFBase(uid_all, iid_all, emb_dim)

        ## 桥
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.capsule_bridge = CapsuleBridge(emb_dim, meta_dim_0, interest_num, wo_capsule=self.wo_capsule).to(device="cuda:0")
        self.public_bridge = nn.Sequential(torch.nn.Linear(emb_dim, emb_dim))
        self.cate_bridge = Cate_bridge(K=self.K, emb_dim=emb_dim)
        self.fuse = torch.nn.Linear(2 * emb_dim, emb_dim)
        self.uid_cluser_labels = None
        self.capsule_mapping_cross_user_embedd_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim))
        self.capsule_mapping_cross_user_embedd_V = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, stage, tau=0.5):
        if stage == "train_src":
            x = self.src_model.forward(x)
            return x
        elif stage in ["train_tgt", "test_tgt"]:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ["train_aug", "test_aug"]:
            x = self.aug_model.forward(x)
            return x

        ##
        ## note:meta_train_stage
        elif stage in ["train_meta"]:

            ##note: 获取基本的内容
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))  ## 获得目标域item embedding
            # src_iid_emb = self.src_model.iid_embedding(x[:, 3].unsqueeze(1))  ## 获得源域item embedding ##没有使用
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))  ## 获得源域 user embedding
            uid_emb_tgt = self.tgt_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))  ## 获得目标域域 user embedding
            rsc_inter_seq_emb = self.src_model.embedding.iid_embedding(x[:, 2:22])  ## 获得源域交互历史item的embedding
            tgt_inter_seq_emb = self.tgt_model.embedding.iid_embedding(x[:, 22:42])  ## 获得目标域交互历史item的embedding
            ###

            ### add:category bridge; aim: 得到类别桥映射 cate_bridge_mapping (B * 110)
            src_uid_cate = self.uid_cluser_labels[x[:, 0]]
            # src_uid_cate = torch.zeros(x[:, 0].shape, dtype=torch.int).to(device="cuda:0")
            cate_bridge_mapping = self.cate_bridge.forward(src_uid_cate)
            ##      note: 基于类别构造的输出 output_cate 和 类别桥目标域user_embedding:uid_emb_cate
            cate_bridge_mapping_w = cate_bridge_mapping[:, : self.emb_dim * self.emb_dim]
            cate_bridge_mapping_W = cate_bridge_mapping_w.view(-1, self.emb_dim, self.emb_dim)  ## mapping ==Fui
            cate_bridge_mapping_b = cate_bridge_mapping[:, self.emb_dim * self.emb_dim :]
            uid_emb_cate = torch.bmm(uid_emb_src, cate_bridge_mapping_W)  # +cate_bridge_mapping_b
            emb_cate = torch.cat([uid_emb_cate, iid_emb], 1)  # 原始生成的目标域embed , 要测试的目标域embed
            output_cate = torch.sum(emb_cate[:, 0, :] * emb_cate[:, 1, :], dim=1)  ##

            ### add:基于公共构造的输出 output_pub 和 uid_emb_pub
            uid_emb_pub = self.public_bridge(uid_emb_src)
            emb_pub = torch.cat([uid_emb_pub, iid_emb], 1)  # 原始生成的目标域embed , 要测试的目标域embed
            output_pub = torch.sum(emb_pub[:, 0, :] * emb_pub[:, 1, :], dim=1)  ##

            ### add: capsule bridge; aim :得到胶囊桥映射 multi_interset_capsule_mapping  B * max_interest* (emb_dim*emb_dim)
            multi_interset_capsule_mapping, gen_mapping, disagree_loss = self.capsule_bridge.forward(rsc_inter_seq_emb, tgt_inter_seq_emb, x[:, 2:])  ##源域交互embedding seq ,目标域交互embedding seq, 和之前两个的总和=> orgin_mapping, gen_mapping
            multi_interset_capsule_mapping = multi_interset_capsule_mapping.view(
                multi_interset_capsule_mapping.shape[0],
                -1,
                self.emb_dim,
                self.emb_dim,
            )  ##
            ##      note:  expanded_uid_emb_src被处理为 (B* max_interest* 1 * emb_dim)
            expanded_uid_emb_src = uid_emb_src.expand(
                size=(multi_interset_capsule_mapping.shape[0], multi_interset_capsule_mapping.shape[1], self.emb_dim),
            ).unsqueeze(dim=2)

            ##      note:  multi_uid_emb_capsule: shape(128,3,10)
            multi_uid_emb_capsule = torch.matmul(expanded_uid_emb_src, multi_interset_capsule_mapping).squeeze(dim=2)  ## Fui的位置
            ##
            ##      做一个全连接映射
            mapped_multi_uid_emb_capsule_K = self.capsule_mapping_cross_user_embedd_K(multi_uid_emb_capsule)
            mapped_ulti_uid_emb_capsule_V = self.capsule_mapping_cross_user_embedd_V(multi_uid_emb_capsule)

            # mapped_ulti_uid_emb_capsule_V=mapped_multi_uid_emb_capsule_K
            ##      note: attention 融合
            multi_uid_emb_capsule_T = mapped_multi_uid_emb_capsule_K.transpose(-1, -2)  ## shape(B,10,max_interest)
            coarse_att = torch.bmm(iid_emb, multi_uid_emb_capsule_T)  ## shape(B,1,max_interest)
            att = torch.softmax(coarse_att, dim=-1)
            ##      note: 基于多胶囊构造的输出 output_capsule 和 胶囊桥目标域user_embedding:uid_emb_capsule
            uid_emb_capsule = torch.bmm(att, mapped_ulti_uid_emb_capsule_V)  ## shape(B,1,10)

            # # ### change: 消融attention融合
            if self.wo_att == 1:
                uid_emb_capsule = mapped_ulti_uid_emb_capsule_V.mean(dim=1, keepdim=True)

            emb_capsule = torch.cat([uid_emb_capsule, iid_emb], 1)  # 生成的目标域embed , 要测试的目标域embed
            output_capsule = torch.sum(emb_capsule[:, 0, :] * emb_capsule[:, 1, :], dim=1)  ##

            ### 对比损失构造
            # cl_loss_func = ContrastiveLoss(batch_size=rsc_inter_seq_emb.size(0), temperature=1, device="cuda:0")
            # cl_loss = cl_loss_func(orgin_mapping, gen_mapping)
            # 对比样本个性化构造输出 gen

            # gen_mapping = gen_mapping.view(-1, self.emb_dim, self.emb_dim)
            # uid_emb_gen = torch.bmm(uid_emb_src, gen_mapping)  ## Fui的位置
            # emb_gen = torch.cat([uid_emb_gen, iid_emb], 1)
            # output_gen = torch.sum(emb_gen[:, 0, :] * emb_gen[:, 1, :], dim=1)  ##

            # # fuse_emb =torch.cat([uid_emb_cate,uid_emb_orgin],dim=-1)

            # proj_fuse_emb =self.fuse(fuse_emb)
            # proj_fuse_emb=uid_emb_cate
            # uid_emb_orgin = proj_fuse_emb

            ##  change: 桥融合
            # uid_emb_fuse= (uid_emb_cate+uid_emb_orgin+uid_emb_pub)/3
            uid_emb_fuse = uid_emb_capsule
            # uid_emb_fuse = uid_emb_tgt

            # 组合uid_emb_fuse 和 item_embedd
            emb_fuse = torch.cat([uid_emb_fuse, iid_emb], 1)  # 原始生成的目标域embed , 要测试的目标域embed
            output_fuse = torch.sum(emb_fuse[:, 0, :] * emb_fuse[:, 1, :], dim=1)  ##

            #  want: 加公式 6的约束
            l2_loss_func = torch.nn.MSELoss()
            l2_loss = l2_loss_func(uid_emb_fuse, uid_emb_tgt)

            # return output_orgin, output_gen, cl_loss,l2_loss

            cl_loss = 0
            l2_loss = 0
            output_gen = output_fuse
            # disgree_loss = 0
            # disagree_loss = disagree_loss
            return output_fuse, output_gen, cl_loss, l2_loss, 0.2 * disagree_loss

        ## note:meta_test_stage
        elif stage in ["test_meta"]:
            ##note: 获取基本的内容
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))  ## 获得目标域item embedding
            # src_iid_emb = self.src_model.iid_embedding(x[:, 3].unsqueeze(1))  ## 获得源域item embedding ##没有使用
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))  ## 获得源域 user embedding
            uid_emb_tgt = self.tgt_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))  ## 获得目标域域 user embedding
            rsc_inter_seq_emb = self.src_model.embedding.iid_embedding(x[:, 2:22])  ## 获得源域交互历史item的embedding
            tgt_inter_seq_emb = self.tgt_model.embedding.iid_embedding(x[:, 22:42])  ## 获得目标域交互历史item的embedding
            ###

            ### add:category bridge; aim: 得到类别桥映射 cate_bridge_mapping (B * 110)
            src_uid_cate = self.uid_cluser_labels[x[:, 0]]
            # src_uid_cate = torch.zeros(x[:, 0].shape, dtype=torch.int).to(device="cuda:0")
            cate_bridge_mapping = self.cate_bridge.forward(src_uid_cate)
            ##      note: 基于类别构造的输出 output_cate 和 类别桥目标域user_embedding:uid_emb_cate
            cate_bridge_mapping_w = cate_bridge_mapping[:, : self.emb_dim * self.emb_dim]
            cate_bridge_mapping_W = cate_bridge_mapping_w.view(-1, self.emb_dim, self.emb_dim)  ## mapping ==Fui
            cate_bridge_mapping_b = cate_bridge_mapping[:, self.emb_dim * self.emb_dim :]
            uid_emb_cate = torch.bmm(uid_emb_src, cate_bridge_mapping_W)  # +cate_bridge_mapping_b
            emb_cate = torch.cat([uid_emb_cate, iid_emb], 1)  # 原始生成的目标域embed , 要测试的目标域embed
            output_cate = torch.sum(emb_cate[:, 0, :] * emb_cate[:, 1, :], dim=1)  ##

            ### add:基于公共构造的输出 output_pub 和 uid_emb_pub
            uid_emb_pub = self.public_bridge(uid_emb_src)
            emb_pub = torch.cat([uid_emb_pub, iid_emb], 1)  # 原始生成的目标域embed , 要测试的目标域embed
            output_pub = torch.sum(emb_pub[:, 0, :] * emb_pub[:, 1, :], dim=1)  ##

            ### add: capsule bridge; aim :得到胶囊桥映射 multi_interset_capsule_mapping  B * max_interest* (emb_dim*emb_dim)
            multi_interset_capsule_mapping = self.capsule_bridge.forward(rsc_inter_seq_emb, None, x[:, 2:], traing_mode=False)  ##源域交互embedding seq ,目标域交互embedding seq, 和之前两个的总和=> orgin_mapping, gen_mapping
            ##      note: multi_interset_capsule_mapping被reshape为 B* max_interest* emb_dim * emb_dim
            multi_interset_capsule_mapping = multi_interset_capsule_mapping.view(multi_interset_capsule_mapping.shape[0], -1, self.emb_dim, self.emb_dim)  ##
            ##      note:  expanded_uid_emb_src被处理为 (B* max_interest* 1 * emb_dim)
            expanded_uid_emb_src = uid_emb_src.expand(
                size=(multi_interset_capsule_mapping.shape[0], multi_interset_capsule_mapping.shape[1], self.emb_dim),
            ).unsqueeze(dim=2)

            ##      note:  multi_uid_emb_capsule: shape(128,3,10)
            multi_uid_emb_capsule = torch.matmul(expanded_uid_emb_src, multi_interset_capsule_mapping).squeeze(dim=2)  ## Fui的位置
            ##      做一个全连接映射
            mapped_multi_uid_emb_capsule_K = self.capsule_mapping_cross_user_embedd_K(multi_uid_emb_capsule)
            mapped_ulti_uid_emb_capsule_V = self.capsule_mapping_cross_user_embedd_V(multi_uid_emb_capsule)
            # mapped_ulti_uid_emb_capsule_V=mapped_multi_uid_emb_capsule_K
            ##      note: attention 融合
            multi_uid_emb_capsule_T = mapped_multi_uid_emb_capsule_K.transpose(-1, -2)  ## shape(B,10,max_interest)
            coarse_att = torch.bmm(iid_emb, multi_uid_emb_capsule_T)  ## shape(B,1,max_interest)
            att = torch.softmax(coarse_att, dim=-1)
            ##      note: 基于多胶囊构造的输出 output_capsule 和 胶囊桥目标域user_embedding:uid_emb_capsule
            uid_emb_capsule = torch.bmm(att, mapped_ulti_uid_emb_capsule_V)  ## shape(B,1,10)

            # # ### change: 消融attention融合
            if self.wo_att == 1:
                uid_emb_capsule = mapped_ulti_uid_emb_capsule_V.mean(dim=1, keepdim=True)

            emb_capsule = torch.cat([uid_emb_capsule, iid_emb], 1)  # 生成的目标域embed , 要测试的目标域embed
            output_capsule = torch.sum(emb_capsule[:, 0, :] * emb_capsule[:, 1, :], dim=1)  ##

            emb_capsule = torch.cat([uid_emb_capsule, iid_emb], 1)  # 原始生成的目标域embed , 要测试的目标域embed
            output_capsule = torch.sum(emb_capsule[:, 0, :] * emb_capsule[:, 1, :], dim=1)  ##

            #### 对比损失构造
            # cl_loss_func = ContrastiveLoss(batch_size=rsc_inter_seq_emb.size(0), temperature=1, device="cuda:0")
            # cl_loss = cl_loss_func(orgin_mapping, gen_mapping)
            # 对比样本个性化构造输出 gen

            # gen_mapping = gen_mapping.view(-1, self.emb_dim, self.emb_dim)
            # uid_emb_gen = torch.bmm(uid_emb_src, gen_mapping)  ## Fui的位置
            # emb_gen = torch.cat([uid_emb_gen, iid_emb], 1)
            # output_gen = torch.sum(emb_gen[:, 0, :] * emb_gen[:, 1, :], dim=1)  ##

            # # fuse_emb =torch.cat([uid_emb_cate,uid_emb_orgin],dim=-1)

            # proj_fuse_emb =self.fuse(fuse_emb)
            # proj_fuse_emb=uid_emb_cate
            # uid_emb_orgin = proj_fuse_emb

            ##  change: 桥融合
            # uid_emb_fuse= (uid_emb_cate+uid_emb_orgin+uid_emb_pub)/3
            uid_emb_fuse = uid_emb_capsule
            # uid_emb_fuse = uid_emb_tgt
            # 组合uid_emb_fuse 和 item_embedd
            emb_fuse = torch.cat([uid_emb_fuse, iid_emb], 1)  # 原始生成的目标域embed , 要测试的目标域embed
            output_fuse = torch.sum(emb_fuse[:, 0, :] * emb_fuse[:, 1, :], dim=1)  ##

            #  want: 加公式 6的约束
            # l2_loss_func = torch.nn.MSELoss()
            # l2_loss = l2_loss_func(output_fuse, uid_emb_tgt)

            # return output_orgin, output_gen, cl_loss,l2_loss
            cl_loss = 0
            l2_loss = 0
            output_gen = output_fuse
            return output_fuse
        elif stage == "train_map":
            src_emb = self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == "test_map":
            uid_emb = self.mapping.forward(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return x.squeeze(1)
