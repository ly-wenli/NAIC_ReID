import torch
import numpy as np
import os
from utils.reranking import re_ranking
from scipy.spatial.distance import cdist

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_( qf, gf.t(),beta=1, alpha=-2)
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

class R1_mAP():
    def __init__(self,cfg, num_query, max_rank=200, feat_norm=True,  reranking=False):
        super(R1_mAP, self).__init__()
        self.cfg = cfg
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pcb_feat = []
        self.img_name_path = []
        self.pcb_feat_split0 = []
        self.pcb_feat_split1 = []
        self.pcb_feat_split2 = []
        self.pcb_feat_split3 = []
        self.pcb_feat_split4 = []
        self.pcb_feat_split5 = []
        self.pcb_feat_split = {}

    # wenli:have local feature use next method
    def update_pcb(self, output):  # called once for each batch
        feat,pcb_feat, imgpath = output
        self.feats.append(feat)
        self.pcb_feat.append(pcb_feat)
        self.img_name_path.extend(imgpath)

    # wenli:have local feature use next method
    def update_pcb_split(self, output):  # called once for each batch
        feat, pcb_feat, imgpath = output
        self.feats.append(feat)
        self.pcb_feat_split0.append(pcb_feat[0])
        self.pcb_feat_split1.append(pcb_feat[1])
        self.pcb_feat_split2.append(pcb_feat[2])
        self.pcb_feat_split3.append(pcb_feat[3])
        self.pcb_feat_split4.append(pcb_feat[4])
        self.pcb_feat_split5.append(pcb_feat[5])
        # for i in range(6):
        #     print("pcb_feat[i].shape",pcb_feat[i].shape)
        #     self.pcb_feat_split[i].append(pcb_feat[i])
        # self.pcb_feat.append(pcb_feat)
        self.img_name_path.extend(imgpath)
    def update_split(self):

        self.pcb_feat_split[0] = self.pcb_feat_split0
        self.pcb_feat_split[1] = self.pcb_feat_split1
        self.pcb_feat_split[2] = self.pcb_feat_split2
        self.pcb_feat_split[3] = self.pcb_feat_split3
        self.pcb_feat_split[4] = self.pcb_feat_split4
        self.pcb_feat_split[5] = self.pcb_feat_split5
    # wenli:have local feature use next method
    def update(self, output):  # called once for each batch
        feat, imgpath = output
        self.feats.append(feat)
        self.img_name_path.extend(imgpath)
    def compute(self,reranking_parameter=[20,6,0.3]):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)

        # if use pcb global feature ,use next method
        if self.cfg.MODEL.IF_USE_PCB:
            if self.cfg.TEST.PCB_GLOBAL_FEAT_ENSEMBLE:
                pcb_feats = torch.cat(self.pcb_feat,dim=0)
                pcb_qf = pcb_feats[:self.num_query]
                pcb_gf = pcb_feats[self.num_query:]


        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_path = self.img_name_path[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_path = self.img_name_path[self.num_query:]
        if self.reranking:
            print('=> Enter reranking')
            print('k1={}, k2={}, lambda_value={}'.format(reranking_parameter[0], reranking_parameter[1],
                                                         reranking_parameter[2]))
            distmat = re_ranking(qf, gf, k1=reranking_parameter[0], k2=reranking_parameter[1], lambda_value=reranking_parameter[2])
            qf = qf.cpu()
            gf = gf.cpu()
            torch.cuda.empty_cache()
            if self.cfg.MODEL.IF_USE_PCB:
                if self.cfg.TEST.PCB_GLOBAL_FEAT_ENSEMBLE:
                    pcb_distmat = re_ranking(pcb_qf, pcb_gf, k1=reranking_parameter[0], k2=reranking_parameter[1], lambda_value=reranking_parameter[2])
                    # del pcb_qf
                    # del pcb_gf
                    # torch.cuda.empty_cache()
                else:
                    if self.cfg.TEST.USE_PCB_MERGE_FEAT:
                        pcb_feats = torch.cat(self.pcb_feat, dim=0)
                        pcb_qf = pcb_feats[:self.num_query]
                        pcb_gf = pcb_feats[self.num_query:]
                        pcb_distmat = re_ranking(pcb_qf, pcb_gf, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                 lambda_value=reranking_parameter[2])
                        ''' 
                        all_pcb_distmat = []
                        pcb_feats = torch.cat(self.pcb_feat, dim=0)

                        pcb_qf = pcb_feats[:self.num_query]
                        pcb_gf = pcb_feats[self.num_query:]
                        print(pcb_qf.shape)
                        m = pcb_qf.shape[0]
                        n = pcb_gf.shape[0]
                        for j in range(m//300+1):
                            temp_pcb_qf = pcb_qf[j * 300:j * 300 + 300]
                            temp_pcb_dist = []
                            for i in range(n // 600 + 1):
                                temp_pcb_gf = pcb_gf[i * 600:i * 600 + 600]
                                pcb_distmat_i = re_ranking(temp_pcb_qf, temp_pcb_gf, k1=reranking_parameter[0],
                                                         k2=reranking_parameter[1],
                                                         lambda_value=reranking_parameter[2])
                                temp_pcb_dist.append(pcb_distmat_i)
                            all_pcb_distmat.append(np.concatenate(temp_pcb_dist,axis=1))
                        pcb_distmat = np.concatenate(all_pcb_distmat, axis=0)
                        '''


                        # for pcb_qf in pcb_qf:
                        #     # print("part pcb_shape",pcb_qf.shape)
                        #     # print("pcb_gf shape is",pcb_gf.shape)
                        #     pcb_qf = torch.unsqueeze(pcb_qf,0)
                        #
                        #     pcb_distmat = re_ranking(pcb_qf, pcb_gf, k1=reranking_parameter[0], k2=reranking_parameter[1],
                        #                              lambda_value=reranking_parameter[2])
                        #     all_pcb_distmat.append(pcb_distmat)
                        # pcb_distmat = np.concatenate(all_pcb_distmat,axis=0)
                        # del pcb_qf
                        # del pcb_gf
                        # torch.cuda.empty_cache()
                    else:
                        pcb_distmat = np.zeros_like(distmat)

                        pcb_feats0 = torch.cat(self.pcb_feat_split0,dim=0)
                        pcb_qf0 = pcb_feats0[:self.num_query]
                        pcb_gf0 = pcb_feats0[self.num_query:]
                        pcb_distmat = re_ranking(pcb_qf0, pcb_gf0, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                    lambda_value=reranking_parameter[2])

                        pcb_feats1 = torch.cat(self.pcb_feat_split1, dim=0)
                        pcb_qf1 = pcb_feats1[:self.num_query]
                        pcb_gf1 = pcb_feats1[self.num_query:]
                        pcb_distmat = pcb_distmat + re_ranking(pcb_qf1, pcb_gf1, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                 lambda_value=reranking_parameter[2])

                        pcb_feats2 = torch.cat(self.pcb_feat_split2, dim=0)
                        pcb_qf2 = pcb_feats2[:self.num_query]
                        pcb_gf2 = pcb_feats2[self.num_query:]
                        pcb_distmat = pcb_distmat + re_ranking(pcb_qf2, pcb_gf2, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                 lambda_value=reranking_parameter[2])

                        pcb_feats3 = torch.cat(self.pcb_feat_split3, dim=0)
                        pcb_qf3 = pcb_feats0[:self.num_query]
                        pcb_gf3 = pcb_feats0[self.num_query:]
                        pcb_distmat = pcb_distmat + re_ranking(pcb_qf3, pcb_gf3, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                 lambda_value=reranking_parameter[2])

                        pcb_feats4 = torch.cat(self.pcb_feat_split4, dim=0)
                        pcb_qf4 = pcb_feats4[:self.num_query]
                        pcb_gf4 = pcb_feats4[self.num_query:]
                        pcb_distmat = pcb_distmat + re_ranking(pcb_qf4, pcb_gf4, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                 lambda_value=reranking_parameter[2])

                        pcb_feats5 = torch.cat(self.pcb_feat_split5, dim=0)
                        pcb_qf5 = pcb_feats5[:self.num_query]
                        pcb_gf5 = pcb_feats5[self.num_query:]
                        pcb_distmat = pcb_distmat + re_ranking(pcb_qf5, pcb_gf5, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                 lambda_value=reranking_parameter[2])
                        """
                        print("self.pcb_feat.shape",np.array(self.pcb_feat_split[0]).shape)
                        for pcb_feat in self.pcb_feat_split:
                            pcb_feats = torch.cat(pcb_feat, dim=0)
                            pcb_qf = pcb_feats[:self.num_query]
                            pcb_gf = pcb_feats[self.num_query:]
                            pcb_distmat = pcb_distmat + re_ranking(pcb_qf, pcb_gf, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                                    lambda_value=reranking_parameter[2])
                        """
                        # del pcb_qf
                        # del pcb_gf
                        # torch.cuda.empty_cache()
        else:
            print('=> Computing DistMat with cosine similarity')
            distmat = cosine_similarity(qf, gf)
            if self.cfg.MODEL.IF_USE_PCB:
                if self.cfg.TEST.PCB_GLOBAL_FEAT_ENSEMBLE:
                    pcb_distmat = cosine_similarity(pcb_qf, pcb_gf)
                else:
                    pcb_distmat = np.zeros_like(distmat)
                    for pcb_feat in self.pcb_feat:
                        pcb_feats = torch.cat(pcb_feat, dim=0)
                        pcb_qf = pcb_feats[:self.num_query]
                        pcb_gf = pcb_feats[self.num_query:]
                        pcb_distmat = pcb_distmat + cosine_similarity(pcb_qf, pcb_gf)
        if self.cfg.MODEL.IF_USE_PCB:
            if self.cfg.TEST.USE_LOCAL:
                distmat = distmat + pcb_distmat
        print(distmat,'distmat')
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        data = dict()
        print(len(g_path), 'self.img_name_q')
        print(len(q_path),'self.img_name_g')
        for q_idx in range(num_q):
            order = indices[q_idx]  # select one row
            result_query = np.array(g_path)[order[:self.max_rank]]
            data[q_path[q_idx]] = [str(i) for i in result_query]
        return data, distmat, q_path, g_path

class R1_mAP_Pseudo():
    def __init__(self, num_query, max_rank=50, feat_norm=True,  reranking=False):
        super(R1_mAP_Pseudo, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.img_name_path = []

    def update(self, output):  # called once for each batch
        feat, imgpath = output
        self.feats.append(feat)
        self.img_name_path.extend(imgpath)

    def compute(self, reranking_parameter=[20, 6, 0.3]):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_path = self.img_name_path[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_path = self.img_name_path[self.num_query:]
        if self.reranking:
            print('=> Enter reranking')
            print('k1={}, k2={}, lambda_value={}'.format(reranking_parameter[0], reranking_parameter[1],
                                                         reranking_parameter[2]))
            distmat = re_ranking(qf, gf, k1=reranking_parameter[0], k2=reranking_parameter[1],
                                 lambda_value=reranking_parameter[2])

        else:
            print('=> Computing DistMat with cosine similarity')
            distmat = cosine_similarity(qf, gf)

        return distmat, q_path, g_path


