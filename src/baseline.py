import torch
import torch.nn.functional as F
import numpy as np
import ot
from util import *
from model_parts import *


def BERTScore_Alignment(vecX, vecY, merge_type):
    vecX = F.normalize(vecX)
    vecY = F.normalize(vecY)
    matrix = torch.matmul(vecX, vecY.t())
    alignments_x = set(['{0}-{1}'.format(i, torch.argmax(matrix[i])) for i in range(matrix.shape[0])])
    alignments_y = set(['{0}-{1}'.format(torch.argmax(matrix[:, i]), i) for i in range(matrix.shape[1])])

    if merge_type == 'intersect':
        alignments = alignments_x & alignments_y
    elif merge_type == 'union':
        alignments = alignments_x | alignments_y
    elif merge_type == 'grow_diag':
        alignments = grow_diag(alignments_x, alignments_y, vecX.shape[0], vecY.shape[0])
    elif merge_type == 'grow_diag_final':
        alignments = grow_diag_final(alignments_x, alignments_y, vecX.shape[0], vecY.shape[0])

    return alignments


class Aligner:
    def __init__(self, ot_type, sinkhorn, chimera, dist_type, weight_type, distortion, thresh, tau, outdir, **kwargs):
        self.ot_type = ot_type
        self.sinkhorn = sinkhorn
        self.chimera = chimera
        self.dist_type = dist_type
        self.weight_type = weight_type
        self.distotion = distortion
        self.thresh = thresh
        self.tau = tau
        self.epsilon = 0.1
        self.stopThr = 1e-6
        self.numItermax = 1000
        self.outdir = outdir
        self.div_type = kwargs['div_type']
        self.save_hyper_parameters()

        self.dist_func = compute_distance_matrix_cosine if dist_type == 'cos' else compute_distance_matrix_l2
        if weight_type == 'uniform':
            self.weight_func = compute_weights_uniform
        else:
            self.weight_func = compute_weights_norm

    def compute_alignment_matrixes(self, s1_vecs, s2_vecs):
        self.align_matrixes = []
        for vecX, vecY in zip(s1_vecs, s2_vecs):
            P = self.compute_optimal_transport(vecX, vecY)
            if torch.is_tensor(P):
                P = P.to('cpu').numpy()

            self.align_matrixes.append(P)

    def get_alignments(self, thresh, assign_cost=False):
        assert len(self.align_matrixes) > 0

        self.thresh = thresh
        all_alignments = []
        for P in self.align_matrixes:
            alignments = self.matrix_to_alignments(P, assign_cost)
            all_alignments.append(alignments)

        return all_alignments

    def matrix_to_alignments(self, P, assign_cost):
        alignments = set()
        align_pairs = np.transpose(np.nonzero(P > self.thresh))
        if assign_cost:
            for i_j in align_pairs:
                alignments.add('{0}-{1}-{2:.4f}'.format(i_j[0], i_j[1], P[i_j[0], i_j[1]]))
        else:
            for i_j in align_pairs:
                alignments.add('{0}-{1}'.format(i_j[0], i_j[1]))

        return alignments

    def compute_optimal_transport(self, s1_word_embeddigs, s2_word_embeddigs):
        s1_word_embeddigs = s1_word_embeddigs.to(torch.float64)
        s2_word_embeddigs = s2_word_embeddigs.to(torch.float64)

        C = self.dist_func(s1_word_embeddigs, s2_word_embeddigs, self.distotion)
        s1_weights, s2_weights = self.weight_func(s1_word_embeddigs, s2_word_embeddigs)

        if self.ot_type == 'ot':
            s1_weights = s1_weights / s1_weights.sum()
            s2_weights = s2_weights / s2_weights.sum()
            s1_weights, s2_weights, C = self.comvert_to_numpy(s1_weights, s2_weights, C)

            if self.sinkhorn:
                P = ot.bregman.sinkhorn_log(s1_weights, s2_weights, C, reg=self.epsilon, stopThr=self.stopThr,
                                            numItermax=self.numItermax)
            else:
                P = ot.emd(s1_weights, s2_weights, C)
            # Min-max normalization
            P = min_max_scaling(P)

        elif self.ot_type == 'pot':
            if self.chimera:
                m = self.tau * self.bertscore_F1(s1_word_embeddigs, s2_word_embeddigs)
                m = min(1.0, m.item())
            else:
                m = self.tau

            s1_weights, s2_weights, C = self.comvert_to_numpy(s1_weights, s2_weights, C)
            m = np.min((np.sum(s1_weights), np.sum(s2_weights))) * m

            if self.sinkhorn:
                P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C,
                                                            reg=self.epsilon,
                                                            m=m, stopThr=self.stopThr, numItermax=self.numItermax)
            else:
                # To cope with round error
                P = ot.partial.partial_wasserstein(s1_weights, s2_weights, C, m=m)
            # Min-max normalization
            P = min_max_scaling(P)

        elif 'uot' in self.ot_type:
            if self.chimera:
                tau = self.tau * self.bertscore_F1(s1_word_embeddigs, s2_word_embeddigs)
            else:
                tau = self.tau

            if self.ot_type == 'uot':
                P = ot.unbalanced.sinkhorn_stabilized_unbalanced(s1_weights, s2_weights, C, reg=self.epsilon, reg_m=tau,
                                                                 stopThr=self.stopThr, numItermax=self.numItermax)
            elif self.ot_type == 'uot-mm':
                P = ot.unbalanced.mm_unbalanced(s1_weights, s2_weights, C, reg_m=tau, div=self.div_type,
                                                stopThr=self.stopThr, numItermax=self.numItermax)
            # Min-max normalization
            P = min_max_scaling(P)

        elif self.ot_type == 'none':
            P = 1 - C

        return P

    def comvert_to_numpy(self, s1_weights, s2_weights, C):
        if torch.is_tensor(s1_weights):
            s1_weights = s1_weights.to('cpu').numpy()
            s2_weights = s2_weights.to('cpu').numpy()
        if torch.is_tensor(C):
            C = C.to('cpu').numpy()

        return s1_weights, s2_weights, C

    # def compute_weights_uniform_pot(self, s1_word_embeddigs, s2_word_embeddigs):
    #     s1_weights = ot.unif(s1_word_embeddigs.shape[0])
    #     s2_weights = ot.unif(s2_word_embeddigs.shape[0])
    #     return s1_weights, s2_weights

    def bertscore_F1(self, vecX, vecY):
        vecX = F.normalize(vecX)
        vecY = F.normalize(vecY)
        matrix = torch.matmul(vecX, vecY.t())

        r = torch.sum(torch.amax(matrix, dim=1)) / matrix.shape[0]
        p = torch.sum(torch.amax(matrix, dim=0)) / matrix.shape[1]
        f = 2 * p * r / (p + r)

        return f.item()

    def save_hyper_parameters(self):
        with open(self.outdir + 'hparams.yaml', 'w') as fw:
            fw.write('ot_type: {0}\n'.format(self.ot_type))
            fw.write('sinkhorn: {0}\n'.format(self.sinkhorn))
            fw.write('epsilon: {0}\n'.format(self.epsilon))
            fw.write('chimera: {0}\n'.format(self.chimera))
            fw.write('dist_type: {0}\n'.format(self.dist_type))
            fw.write('weight_type: {0}\n'.format(self.weight_type))
            fw.write('div_type: {0}\n'.format(self.div_type))
            fw.write('tau: {0}\n'.format(self.tau))
            fw.write('threshold: {0}\n'.format(self.thresh))
