import torch
from torch import nn
import transformers
import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from util import *
from model_parts import *


class NeuralAlignerBase(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.sinkhorn_epsilon = 0.1
        self.stopThr = 1e-4
        self.numItermax = 1000
        self.ot_type = kwargs['ot_type']

        self.loss_fnc = nn.BCELoss(reduction='sum')
        self.positive_label = 1
        self.negative_label = 0

    def get_word_embeddings(self, hidden_tensors, subword_to_word_conv, s2_start_idx):
        word_num = torch.amax(subword_to_word_conv).item() + 1
        word_embeddings = torch.vstack(
            ([torch.mean(hidden_tensors[subword_to_word_conv == word_idx], dim=0) for word_idx in
              range(word_num)]))

        s1_word_embeddigs = word_embeddings[0:s2_start_idx, :]
        s2_word_embeddigs = word_embeddings[s2_start_idx:, :]

        return s1_word_embeddigs, s2_word_embeddigs

    def get_attention_per_word(self, attentions, subword_to_word_conv, s2_start_idx):
        eps = 1e-12
        word_num = torch.amax(subword_to_word_conv).item() + 1
        attentions = torch.vstack(
            ([torch.sum(attentions[subword_to_word_conv == word_idx], dim=0) for word_idx in
              range(word_num)]))
        attentions = torch.vstack(
            ([torch.sum(attentions[:, subword_to_word_conv == word_idx], dim=1) for word_idx in
              range(word_num)]))

        attentions = attentions + eps
        atten_s2t = attentions[0:s2_start_idx, s2_start_idx:word_num]
        atten_s2t = (atten_s2t.T / atten_s2t.sum(1)).T
        atten_t2s = attentions[s2_start_idx:word_num, 0:s2_start_idx]
        atten_t2s = (atten_t2s.T / atten_t2s.sum(1)).T
        atten = (atten_s2t + atten_t2s.T) / 2

        return atten

    def convert_to_word_embeddings(self, offset_mapping, token_ids, hidden_tensors, pooling):
        pooling_method = torch.mean if pooling == 'mean' else torch.amax
        word_idx = -1
        subword_to_word_conv = np.full((hidden_tensors.shape[0]), -1)
        # Bug in hugging face tokenizer? Sometimes Metaspace is inserted
        metaspace = getattr(self.tokenizer.decoder, "replacement", None)
        metaspace = self.tokenizer.decoder.prefix if metaspace is None else metaspace
        tokenizer_bug_idxes = [i for i, x in enumerate(self.tokenizer.convert_ids_to_tokens(token_ids)) if
                               x == metaspace]

        for subw_idx, offset in enumerate(offset_mapping):
            if subw_idx in tokenizer_bug_idxes:
                continue
            elif offset[0] == offset[1]:  # Special token
                continue
            elif offset[0] == 0:
                word_idx += 1
                subword_to_word_conv[subw_idx] = word_idx
            else:
                subword_to_word_conv[subw_idx] = word_idx

        word_embeddings = torch.vstack(
            ([pooling_method(hidden_tensors[subword_to_word_conv == word_idx], dim=0) for word_idx in
              range(word_idx + 1)]))

        sep_tok_indices = [i for i, x in enumerate(token_ids) if x == self.tokenizer.sep_token_id]
        s2_start_idx = subword_to_word_conv[
            sep_tok_indices[0] + np.argmax(subword_to_word_conv[sep_tok_indices[0]:] > -1)]

        s1_word_embeddigs = word_embeddings[0:s2_start_idx, :]
        s2_word_embeddigs = word_embeddings[s2_start_idx:, :]

        return s1_word_embeddigs, s2_word_embeddigs

    def convert_to_word_attentions(self, offset_mapping, token_ids, attentions, cross_attention):
        word_idx = -1
        subword_to_word_conv = np.full((attentions.shape[0]), -1)
        # Bug in hugging face tokenizer? Sometimes Metaspace is inserted
        metaspace = getattr(self.tokenizer.decoder, "replacement", None)
        metaspace = self.tokenizer.decoder.prefix if metaspace is None else metaspace
        tokenizer_bug_idxes = [i for i, x in enumerate(self.tokenizer.convert_ids_to_tokens(token_ids)) if
                               x == metaspace]

        for subw_idx, offset in enumerate(offset_mapping):
            if subw_idx in tokenizer_bug_idxes:
                continue
            elif offset[0] == offset[1]:  # Special token
                continue
            elif offset[0] == 0:
                word_idx += 1
                subword_to_word_conv[subw_idx] = word_idx
            else:
                subword_to_word_conv[subw_idx] = word_idx

        # Take mean for 'from', sum for 'to' following https://arxiv.org/pdf/1906.04341.pdf
        attentions = torch.vstack(
            ([torch.mean(attentions[subword_to_word_conv == word_idx], dim=0) for word_idx in range(word_idx + 1)]))
        attentions = torch.vstack(
            ([torch.sum(attentions[:, subword_to_word_conv == word_idx], dim=1) for word_idx in range(word_idx + 1)]))

        sep_tok_indices = [i for i, x in enumerate(token_ids) if x == self.tokenizer.sep_token_id]
        s2_start_idx = subword_to_word_conv[
            sep_tok_indices[0] + np.argmax(subword_to_word_conv[sep_tok_indices[0]:] > -1)]

        if cross_attention:
            s1_attn = attentions[0:s2_start_idx, s2_start_idx:]
            s2_attn = attentions[s2_start_idx:, 0:s2_start_idx]

        else:
            # self-attention: average 'from' and 'to' attention weights
            s1_attn = (attentions[0:s2_start_idx, 0:s2_start_idx] + attentions[0:s2_start_idx, 0:s2_start_idx].t()) / 2
            s2_attn = (attentions[s2_start_idx:, s2_start_idx:] + attentions[s2_start_idx:, s2_start_idx:].t()) / 2

        return s1_attn, s2_attn

    def batch_center_embedding(self, hidden_outputs):
        mean_vec = torch.mean(hidden_outputs.detach(), dim=(1, 0))
        hidden_outputs = hidden_outputs - mean_vec
        return hidden_outputs

    def compute_loss(self, batch, forward_outputs):
        loss = 0
        transport_matrix_list = forward_outputs['transport_matrix_list']
        for bidx in range(len(transport_matrix_list)):
            P = transport_matrix_list[bidx]
            y = torch.full_like(P, fill_value=self.negative_label, device=self.device)
            # weights = torch.ones_like(P, device=self.device)
            for pair in batch['atuples'][bidx]:
                y[pair[0], pair[1]] = self.positive_label

            loss += self.loss_fnc(P, y)

        loss = loss / len(transport_matrix_list)
        logs = {"loss": loss}
        return loss, logs

    def compute_score(self, pred, gold):
        gold = set(gold.split())
        if len(pred) > 0:
            precision = len(gold & pred) / len(pred)
        else:
            if len(gold) == 0:
                precision = 1
            else:
                precision = 0
        if len(gold) > 0:
            recall = len(gold & pred) / len(gold)
        else:
            if len(pred) == 0:
                recall = 1
            else:
                recall = 0
        if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
            acc = 1
        else:
            acc = 0

        return precision, recall, acc

    def compute_null_score(self, pred, gold, sent1, sent2):
        gold = set(gold.split())
        sent1 = sent1.split()
        sent2 = sent2.split()
        gold_s1_indices, gold_s2_indices = get_aligned_indices(gold)
        gold_null_s1_indices = set(range(len(sent1))) - gold_s1_indices
        gold_null_s2_indices = set(range(len(sent2))) - gold_s2_indices

        aligned_s1_indices, aligned_s2_indices = get_aligned_indices(pred)
        pred_null_s1_indices = set(range(len(sent1))) - aligned_s1_indices
        pred_null_s2_indices = set(range(len(sent2))) - aligned_s2_indices

        if len(pred_null_s1_indices) + len(pred_null_s2_indices) > 0:
            precision = (len(gold_null_s1_indices & pred_null_s1_indices) + len(
                gold_null_s2_indices & pred_null_s2_indices)) / (len(pred_null_s1_indices) + len(pred_null_s2_indices))
        else:
            if len(gold_null_s1_indices) + len(gold_null_s2_indices) == 0:
                precision = 1
            else:
                precision = 0
        if len(gold_null_s1_indices) + len(gold_null_s2_indices) > 0:
            recall = (len(gold_null_s1_indices & pred_null_s1_indices) + len(
                gold_null_s2_indices & pred_null_s2_indices)) / (len(gold_null_s1_indices) + len(gold_null_s2_indices))
        else:
            if len(pred_null_s1_indices) + len(pred_null_s2_indices) == 0:
                recall = 1
            else:
                recall = 0

        if len(gold_null_s1_indices & pred_null_s1_indices) == len(gold_null_s1_indices) and len(
                gold_null_s1_indices & pred_null_s1_indices) == len(pred_null_s1_indices) and len(
            gold_null_s2_indices & pred_null_s2_indices) == len(gold_null_s2_indices) and len(
            gold_null_s2_indices & pred_null_s2_indices) == len(pred_null_s2_indices):
            acc = 1.0
        else:
            acc = 0.0

        return precision, recall, acc

    def compute_total_score(self, pred, gold, sent1, sent2):
        gold = set(gold.split())
        sent1 = sent1.split()
        sent2 = sent2.split()
        gold_s1_indices, gold_s2_indices = get_aligned_indices(gold)
        gold_null_s1_indices = set(range(len(sent1))) - gold_s1_indices
        gold_null_s2_indices = set(range(len(sent2))) - gold_s2_indices

        aligned_s1_indices, aligned_s2_indices = get_aligned_indices(pred)
        pred_null_s1_indices = set(range(len(sent1))) - aligned_s1_indices
        pred_null_s2_indices = set(range(len(sent2))) - aligned_s2_indices

        precision = (len(gold & pred) + len(gold_null_s1_indices & pred_null_s1_indices) + len(
            gold_null_s2_indices & pred_null_s2_indices)) / (
                            len(pred) + len(pred_null_s1_indices) + len(pred_null_s2_indices))
        recall = (len(gold & pred) + len(gold_null_s1_indices & pred_null_s1_indices) + len(
            gold_null_s2_indices & pred_null_s2_indices)) / (
                         len(gold) + len(gold_null_s1_indices) + len(gold_null_s2_indices))

        if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
            acc = 1
        else:
            acc = 0

        return precision, recall, acc

    def evaluate_alignments(self, outputs, val_or_test, save_prediction=False):
        precision_all = []
        recall_all = []
        accuracy_all = []
        null_precision_all = []
        null_recall_all = []
        null_accuracy_all = []
        total_precision_all = []
        total_recall_all = []
        total_accuracy_all = []
        # align_thresh = self.determine_align_thresh(outputs)
        for output in outputs:
            transport_matrix_list = output['transport_matrix_list']
            golds = output['gold']
            sents1 = output['s1_sents']
            sents2 = output['s2_sents']
            for bidx, P in enumerate(transport_matrix_list):
                alignments = set()
                align_pairs = np.transpose(np.nonzero(P > self.align_hypr))
                for i_j in align_pairs:
                    alignments.add('{0}-{1}'.format(i_j[0], i_j[1]))

                p, r, acc = self.compute_score(alignments, golds[bidx])
                null_p, null_r, null_acc = self.compute_null_score(alignments, golds[bidx], sents1[bidx], sents2[bidx])
                total_p, total_r, total_acc = self.compute_total_score(alignments, golds[bidx], sents1[bidx],
                                                                       sents2[bidx])
                precision_all.append(p)
                recall_all.append(r)
                accuracy_all.append(acc)
                null_precision_all.append(null_p)
                null_recall_all.append(null_r)
                null_accuracy_all.append(null_acc)
                total_precision_all.append(total_p)
                total_recall_all.append(total_r)
                total_accuracy_all.append(total_acc)

        precision = sum(precision_all) / len(precision_all)
        recall = sum(recall_all) / len(recall_all)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        accuracy = sum(accuracy_all) / len(accuracy_all)

        null_precision = sum(null_precision_all) / len(null_precision_all)
        null_recall = sum(null_recall_all) / len(null_recall_all)
        if null_precision + null_recall > 0:
            null_f1 = 2 * null_precision * null_recall / (null_precision + null_recall)
        else:
            null_f1 = 0.0
        null_accuracy = sum(null_accuracy_all) / len(null_accuracy_all)

        total_precision = sum(total_precision_all) / len(total_precision_all)
        total_recall = sum(total_recall_all) / len(total_recall_all)
        if total_precision + total_recall > 0:
            total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
        else:
            total_f1 = 0.0
        total_accuracy = sum(total_accuracy_all) / len(total_accuracy_all)

        logs = {"precision": precision, "recall": recall, "f1": f1, "exact_match": accuracy,
                "null_precision": null_precision, "null_recall": null_recall, "null_f1": null_f1,
                "null_exact_match": null_accuracy, "total_precision": total_precision, "total_recall": total_recall,
                "total_f1": total_f1, "total_exact_match": total_accuracy}

        if save_prediction:
            with open(self.logger.log_dir + '/{0}_{1}_report.txt'.format(self.data, val_or_test), 'a') as fw:
                fw.write('Alignment Hyperparameter:\t{0:.4f}\n'.format(self.align_hypr))
                fw.write(
                    'Precision\tRecall\tF1\tExactMatch\t[Null]Precision\t[Null]Recall\t[Null]F1\t[Null]ExactMatch\t[Total]Precision\t[Total]Recall\t[Total]F1\t[Total]ExactMatch\n')
                fw.write(
                    '{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4f}\t{11:.4f}\n'.format(
                        precision * 100,
                        recall * 100,
                        f1 * 100,
                        accuracy * 100,
                        null_precision * 100,
                        null_recall * 100,
                        null_f1 * 100,
                        null_accuracy * 100,
                        total_precision * 100,
                        total_recall * 100,
                        total_f1 * 100,
                        total_accuracy * 100, ))

            with open(self.logger.log_dir + '/{0}_{1}_alignments_{2:.4f}.txt'.format(self.data, val_or_test,
                                                                                     self.align_hypr),
                      'w') as fw:
                fw.write('Sentence_1\tSentence_2\tGold\tPrediction\n')
                for output in outputs:
                    transport_matrix_list = output['transport_matrix_list']
                    golds = output['gold']
                    s1_sents = output['s1_sents']
                    s2_sents = output['s2_sents']
                    for bidx, P in enumerate(transport_matrix_list):
                        alignments = set()
                        align_pairs = np.transpose(np.nonzero(P > self.align_hypr))
                        for i_j in align_pairs:
                            alignments.add('{0}-{1}-{2:.4f}'.format(i_j[0], i_j[1], P[i_j[0], i_j[1]]))

                        fw.write('{0}\t{1}\t{2}\t{3}\n'.format(s1_sents[bidx], s2_sents[bidx], golds[bidx],
                                                               ' '.join(alignments)))

        return logs

    def determine_sinkhorn_threshold(self, outputs, val_or_test):
        best_thresh = 0.0
        best_val_log = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "null_precision": 0.0,
                        "null_recall": 0.0, "null_f1": 0.0, "null_exact_match": 0.0, "total_precision": 0.0,
                        "total_recall": 0.0, "total_f1": 0.0, "total_exact_match": 0.0}

        if self.ot_type == 'uot-mm':  # No thresholding is necessary
            self.align_hypr = best_thresh = 0.0
            best_val_log = self.evaluate_alignments(outputs, val_or_test, save_prediction=False)
        else:
            # Determine threshold
            for th in np.linspace(0.0, 1.0, num=100, endpoint=True):
                self.align_hypr = th
                logs = self.evaluate_alignments(outputs, val_or_test, save_prediction=False)
                if logs['total_f1'] > best_val_log['total_f1']:
                    best_thresh = th
                    best_val_log = logs
        return best_thresh, best_val_log

    def training_step(self, batch, batch_idx):
        forward_outputs = self.forward(batch)
        loss, logs = self.compute_loss(batch, forward_outputs)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        self.log_dict({f"batch_size": self.batch_size})
        return loss

    def bidirectional_align(self, batch):
        outputs = self.forward(batch)
        transport_matrix_list = [matrix.cpu().detach().numpy() for matrix in outputs['transport_matrix_list']]
        return transport_matrix_list

    def validation_step(self, batch, batch_idx):
        transport_matrix_list = self.bidirectional_align(batch)

        return {'transport_matrix_list': transport_matrix_list, 'gold': batch['alignments'],
                's1_sents': batch['s1_sents'], 's2_sents': batch['s2_sents']}

    def validation_epoch_end(self, outputs):
        best_thresh, _ = self.determine_sinkhorn_threshold(outputs, 'val')
        self.align_hypr = best_thresh
        logs = self.evaluate_alignments(outputs, 'val', save_prediction=False)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        self.log_dict({f"batch_size": self.batch_size})
        return logs

    def on_test_start(self):
        # Determine best align_hypr
        # generate alignments
        eval_dict = {'transport_matrix_list': [], 'gold': [], 's1_sents': [], 's2_sents': []}
        all_transport_matrices = []
        for batch in tqdm.tqdm(self.val_dataloader(), desc='Encoding validation set'):
            batch = send_batch_to_gpu(batch, self.device)
            with torch.no_grad():
                all_transport_matrices += self.bidirectional_align(batch)

            eval_dict['gold'] += batch['alignments']
            eval_dict['s1_sents'] += batch['s1_sents']
            eval_dict['s2_sents'] += batch['s2_sents']
        eval_dict['transport_matrix_list'] = all_transport_matrices
        best_thresh, best_val_log = self.determine_sinkhorn_threshold([eval_dict], 'val')
        self.align_hypr = best_thresh
        logs = self.evaluate_alignments([eval_dict], 'val', save_prediction=True)

    def test_step(self, batch, batch_idx):
        transport_matrix_list = self.bidirectional_align(batch)

        return {'transport_matrix_list': transport_matrix_list, 'gold': batch['alignments'],
                's1_sents': batch['s1_sents'], 's2_sents': batch['s2_sents']}

    def test_epoch_end(self, outputs):
        logs = self.evaluate_alignments(outputs, 'test', save_prediction=True)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        self.log_dict({f"batch_size": self.batch_size})

    def configure_optimizers(self):
        optimizer = transformers.Adafactor(self.parameters(), scale_parameter=False, relative_step=False,
                                           warmup_init=False, lr=self.lr)
        return optimizer

    # return the dataloader for each split
    def train_dataloader(self):
        return self.data_loader(self.data, 'train')

    def val_dataloader(self):
        return self.data_loader(self.data, 'dev')

    def test_dataloader(self):
        return self.data_loader(self.data, 'test')

    def data_loader(self, data, type):
        sents1, sents2, align_lists = load_WA_corpus(data, type, self.sure_and_possible)

        if type == 'train':
            shuffle = True
        else:
            shuffle = False

        input_s2t = self.tokenizer(text=sents1, text_pair=sents2, padding=True, truncation=True,
                                   is_split_into_words=True,
                                   return_offsets_mapping=True, return_tensors="pt")

        all_subword_to_word_map, all_s2_start_idx = get_token_word_mapping(input_s2t, self.tokenizer)

        return DataLoader(
            AlignmentDataset(sents1, sents2, input_s2t, align_lists, all_subword_to_word_map, all_s2_start_idx),
            batch_size=self.batch_size,
            shuffle=shuffle, collate_fn=alignment_collate)
