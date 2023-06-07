import torch
import torch.nn.functional as F
from util import min_max_scaling

def compute_distance_matrix_cosine(s1_word_embeddigs, s2_word_embeddigs, distortion_ratio):
    C = (torch.matmul(F.normalize(s1_word_embeddigs), F.normalize(s2_word_embeddigs).t()) + 1.0) / 2  # Range 0-1
    C = apply_distortion(C, distortion_ratio)
    C = min_max_scaling(C)  # Range 0-1
    C = 1.0 - C  # Convert to distance

    return C


def compute_distance_matrix_l2(s1_word_embeddigs, s2_word_embeddigs, distortion_ratio):
    C = torch.cdist(s1_word_embeddigs, s2_word_embeddigs, p=2)
    C = min_max_scaling(C)  # Range 0-1
    C = 1.0 - C  # Convert to similarity
    C = apply_distortion(C, distortion_ratio)
    C = min_max_scaling(C)  # Range 0-1
    C = 1.0 - C  # Convert to distance

    return C


def apply_distortion(sim_matrix, ratio):
    shape = sim_matrix.shape
    if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
        return sim_matrix

    pos_x = torch.tensor([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])],
                         device='cuda')
    pos_y = torch.tensor([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])],
                         device='cuda')
    distortion_mask = 1.0 - ((pos_x - pos_y.T) ** 2) * ratio

    sim_matrix = torch.mul(sim_matrix, distortion_mask)

    return sim_matrix


def compute_weights_norm(s1_word_embeddigs, s2_word_embeddigs):
    s1_weights = torch.norm(s1_word_embeddigs, dim=1)
    s2_weights = torch.norm(s2_word_embeddigs, dim=1)
    return s1_weights, s2_weights


def compute_weights_uniform(s1_word_embeddigs, s2_word_embeddigs):
    s1_weights = torch.ones(s1_word_embeddigs.shape[0], dtype=torch.float64, device='cuda')
    s2_weights = torch.ones(s2_word_embeddigs.shape[0], dtype=torch.float64, device='cuda')

    # # Uniform weights to make L2 norm=1
    # s1_weights /= torch.linalg.norm(s1_weights)
    # s2_weights /= torch.linalg.norm(s2_weights)

    return s1_weights, s2_weights
