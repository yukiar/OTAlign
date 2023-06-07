import ot
from transformers import AutoTokenizer, AutoModel
from torch import nn
from model_parts import *
from my_pot import entropic_partial_wasserstein
from model_base import NeuralAlignerBase


class NeuralAligner(NeuralAlignerBase):
    def __init__(self, pretrained_model, data, sure_and_possible, batch_size, lr, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.pretrained_model = pretrained_model
        self.data = data
        self.sure_and_possible = sure_and_possible
        self.batch_size = batch_size
        self.lr = lr

        # Encoder
        self.encoder = AutoModel.from_pretrained(pretrained_model, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, model_max_length=self.encoder.config.max_position_embeddings)

        # OT settings
        self.align_hypr = -1
        self.ot_hyp = kwargs['ot_hyp']
        self.distortion = kwargs['distortion']
        self.div_type = kwargs['div_type']
        self.use_attention_map = kwargs['attention']
        self.dist_func = compute_distance_matrix_cosine if kwargs[
                                                               'dist_type'] == 'cos' else compute_distance_matrix_l2
        self.weight_func = compute_weights_norm if kwargs['weight_type'] == 'norm' else compute_weights_uniform

        self.vec_transform = nn.Sequential(nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
                                           nn.Tanh())

    def forward(self, inputs):
        if self.ot_type == 'none':
            return self.pretrain_encoder(inputs)
        elif self.ot_type == 'uot':
            return self.forward_uot(inputs)
        elif self.ot_type == 'pot':
            return self.forward_pot(inputs)
        else:
            return self.forward_ot(inputs)

    def pretrain_encoder(self, inputs):
        transport_matrix_list = []
        s2t_outputs = self.encoder(inputs['s2t_input_ids'], inputs['s2t_attention_mask'],
                                   inputs['s2t_token_type_ids'])

        token_embeddings = self.vec_transform(s2t_outputs[0])
        attentions = s2t_outputs[2][-1]  # Attention of the last layer

        for bidx in range(inputs['s2t_input_ids'].shape[0]):
            C, _, _ = self.compute_cost(token_embeddings[bidx], attentions[bidx], self.distortion,
                                        inputs['subword_to_word_map'][bidx],
                                        inputs['s2_start_idx'][bidx])
            C = torch.ones_like(C) - C

            transport_matrix_list.append(C.to(torch.float32))

        return {'transport_matrix_list': transport_matrix_list}

    def forward_uot(self, inputs):
        transport_matrix_list = []
        s2t_outputs = self.encoder(inputs['s2t_input_ids'], inputs['s2t_attention_mask'],
                                   inputs['s2t_token_type_ids'])

        token_embeddings = self.vec_transform(s2t_outputs[0])
        attentions = s2t_outputs[2][-1]  # Attention of the last layer

        for bidx in range(inputs['s2t_input_ids'].shape[0]):
            C, s1_word_embeddigs, s2_word_embeddigs = self.compute_cost(token_embeddings[bidx], attentions[bidx],
                                                                        self.distortion,
                                                                        inputs['subword_to_word_map'][bidx],
                                                                        inputs['s2_start_idx'][bidx])
            s1_weights, s2_weights = self.weight_func(s1_word_embeddigs, s2_word_embeddigs)

            # Compute unbalanced OT
            P = ot.unbalanced.sinkhorn_stabilized_unbalanced(s1_weights, s2_weights, C,
                                                             reg=self.sinkhorn_epsilon, reg_m=self.ot_hyp,
                                                             stopThr=self.stopThr, numItermax=self.numItermax)
            P = min_max_scaling(P)
            transport_matrix_list.append(P.to(torch.float32))

        return {'transport_matrix_list': transport_matrix_list}

    def forward_pot(self, inputs):
        transport_matrix_list = []
        s2t_outputs = self.encoder(inputs['s2t_input_ids'], inputs['s2t_attention_mask'],
                                   inputs['s2t_token_type_ids'])

        token_embeddings = self.vec_transform(s2t_outputs[0])
        attentions = s2t_outputs[2][-1]  # Attention of the last layer

        for bidx in range(inputs['s2t_input_ids'].shape[0]):
            C, s1_word_embeddigs, s2_word_embeddigs = self.compute_cost(token_embeddings[bidx], attentions[bidx],
                                                                        self.distortion,
                                                                        inputs['subword_to_word_map'][bidx],
                                                                        inputs['s2_start_idx'][bidx])
            s1_weights, s2_weights = self.weight_func(s1_word_embeddigs, s2_word_embeddigs)

            # Compute partial OT
            m = self.ot_hyp * torch.minimum(torch.sum(s1_weights.detach().clone()),
                                            torch.sum(s2_weights.detach().clone()))

            P = entropic_partial_wasserstein(s1_weights, s2_weights, C,
                                             reg=self.sinkhorn_epsilon, m=m, stopThr=self.stopThr,
                                             numItermax=self.numItermax)
            P = min_max_scaling(P)
            transport_matrix_list.append(P.to(torch.float32))

        return {'transport_matrix_list': transport_matrix_list}

    def forward_ot(self, inputs):
        transport_matrix_list = []
        s2t_outputs = self.encoder(inputs['s2t_input_ids'], inputs['s2t_attention_mask'],
                                   inputs['s2t_token_type_ids'])

        token_embeddings = self.vec_transform(s2t_outputs[0])
        attentions = s2t_outputs[2][-1]  # Attention of the last layer

        for bidx in range(inputs['s2t_input_ids'].shape[0]):
            C, s1_word_embeddigs, s2_word_embeddigs = self.compute_cost(token_embeddings[bidx], attentions[bidx],
                                                                        self.distortion,
                                                                        inputs['subword_to_word_map'][bidx],
                                                                        inputs['s2_start_idx'][bidx])
            s1_weights, s2_weights = self.weight_func(s1_word_embeddigs, s2_word_embeddigs)

            s1_weights = s1_weights / s1_weights.sum()
            s2_weights = s2_weights / s2_weights.sum()

            # Compute OT
            P = ot.bregman.sinkhorn_log(s1_weights, s2_weights, C, reg=self.sinkhorn_epsilon, stopThr=self.stopThr,
                                        numItermax=self.numItermax)
            P = min_max_scaling(P)
            transport_matrix_list.append(P.to(torch.float32))

        return {'transport_matrix_list': transport_matrix_list}

    def compute_cost(self, token_embeddings, attentions, distortion, subword_to_word_map, s2_start_idx):
        # Estimate parameters for unbalanced OT
        s1_word_embeddigs, s2_word_embeddigs = self.get_word_embeddings(
            token_embeddings, subword_to_word_map, s2_start_idx)

        s1_word_embeddigs = s1_word_embeddigs.to(torch.float64)
        s2_word_embeddigs = s2_word_embeddigs.to(torch.float64)

        # Estimate cost matrix
        # get embeddings to well represent semantics
        C = self.dist_func(s1_word_embeddigs, s2_word_embeddigs, distortion)

        if self.use_attention_map:
            attention_weights = attentions.mean(0)  # mean of all attention heads
            attention_weights = self.get_attention_per_word(attention_weights, subword_to_word_map, s2_start_idx)
            attention_weights = 1.0 - attention_weights.to(torch.float64)
            C = (C + attention_weights) / 2

        return C, s1_word_embeddigs, s2_word_embeddigs
