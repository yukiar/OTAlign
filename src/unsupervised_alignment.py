import argparse, random, os
import glob

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from util import *
from baseline import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='mtref', type=str,
                    choices=['mtref', 'wiki', 'newsela', 'arxiv', 'msr', 'edinburgh'], required=True)
parser.add_argument('--sure_and_possible', action='store_true')
parser.add_argument('--out', default='../out/unsupervised/', type=str, required=True)
parser.add_argument('--model', default='bert-base-cased', type=str)
parser.add_argument('--ot_type', default='OT type', type=str, required=True, nargs='*')
parser.add_argument('--weight_type', help='Weight type', type=str, choices=['norm', 'uniform', '--'], default='--')
parser.add_argument('--dist_type', help='Distance metric', type=str, choices=['cos', 'l2'], required=True)
parser.add_argument('--sinkhorn', help='Use sinkhorn', action='store_true')
parser.add_argument('--div_type', help='uot_mm divergence', type=str, default='--', choices=['kl', 'l2', '--'])
parser.add_argument('--chimera', help='Use BERTScore for parameter estimation', action='store_true')
parser.add_argument('--pair_encode', action='store_true')
parser.add_argument('--centering', action='store_true')
parser.add_argument('--layer', default=-3, type=int, help='Which hidden layer to use as token embedding')
parser.add_argument('--seed', help='number of attention heads', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def evaluate(golds, predictions, data_type, out_path):
    precision_all = []
    recall_all = []
    acc = []

    for gold, pred in zip(golds, predictions):
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
        precision_all.append(precision)
        recall_all.append(recall)
        if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
            acc.append(1)
        else:
            acc.append(0)
    precision = sum(precision_all) / len(precision_all)
    recall = sum(recall_all) / len(recall_all)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    accuracy = sum(acc) / len(acc)

    # print('Task: {0} {1}'.format(args.data, args.data_type))
    # print('Precision\t{0:.5f}'.format(precision * 100))
    # print('Recall\t{0:.5f}'.format(recall * 100))
    # print('F1\t{0:.5f}'.format(f1 * 100))
    # print('ExactMatch\t{0:.5f}'.format(accuracy * 100))

    with open(out_path, 'a') as fw:
        fw.write('Task: {0} {1}\n'.format(args.data, data_type))
        fw.write('Precision\tRecall\tF1\tExactMatch\n')
        fw.write('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}\n'.format(precision * 100, recall * 100, f1 * 100, accuracy * 100))

    return {'precision': precision, 'recall': recall, 'f1': f1, 'exact_match': accuracy}


def eval_null_alignments(golds, sents1, sents2, predictions, data_type, out_path):
    null_precision, null_recall, null_EM = [], [], []
    null_ratio_corpus, null_ratio_sent = 0, 0
    for gold, sent1, sent2, pred in zip(golds, sents1, sents2, predictions):
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
            null_ratio_corpus += 1
            null_ratio_sent += (len(gold_null_s1_indices) + len(gold_null_s2_indices)) / (len(sent1) + len(sent2))
        else:
            if len(pred_null_s1_indices) + len(pred_null_s2_indices) == 0:
                recall = 1
            else:
                recall = 0
        null_precision.append(precision)
        null_recall.append(recall)
        if len(gold_null_s1_indices & pred_null_s1_indices) == len(gold_null_s1_indices) and len(
                gold_null_s1_indices & pred_null_s1_indices) == len(pred_null_s1_indices) and len(
            gold_null_s2_indices & pred_null_s2_indices) == len(gold_null_s2_indices) and len(
            gold_null_s2_indices & pred_null_s2_indices) == len(pred_null_s2_indices):
            null_EM.append(1)
        else:
            null_EM.append(0)

    precision = sum(null_precision) / len(null_precision)
    recall = sum(null_recall) / len(null_recall)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    accuracy = sum(null_EM) / len(null_EM)

    # print('Task [NULL]: {0} {1}'.format(args.data, args.data_type))
    # print('Null ratio: {0:.4f}'.format(null_ratio / len(golds)))
    # print('Precision\t{0:.5f}'.format(precision * 100))
    # print('Recall\t{0:.5f}'.format(recall * 100))
    # print('F1\t{0:.5f}'.format(f1 * 100))
    # print('ExactMatch\t{0:.5f}'.format(accuracy * 100))

    with open(out_path, 'a') as fw:
        fw.write('*******************\n')
        fw.write('Task [NULL]: {0} {1}\n'.format(args.data, data_type))
        fw.write('Null ratio (corpus): {0:.4f}\n'.format(null_ratio_corpus / len(golds)))
        fw.write('Null ratio (sentence): {0:.4f}\n'.format(null_ratio_sent / len(golds)))
        fw.write('Precision\tRecall\tF1\tExactMatch\n')
        fw.write('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}\n'.format(precision * 100, recall * 100, f1 * 100, accuracy * 100))
        fw.write('*******************\n')

    return {'null_precision': precision, 'null_recall': recall, 'null_f1': f1, 'null_exact_match': accuracy}


def evaluate_total_score(golds, predictions, sents1, sents2, data_type, out_path):
    precision_all = []
    recall_all = []
    acc = []

    for gold, pred, sent1, sent2 in zip(golds, predictions, sents1, sents2):
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

        precision_all.append(precision)
        recall_all.append(recall)

        if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
            acc.append(1)
        else:
            acc.append(0)

    precision = sum(precision_all) / len(precision_all)
    recall = sum(recall_all) / len(recall_all)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    accuracy = sum(acc) / len(acc)

    # print('Task: {0} {1}'.format(args.data, args.data_type))
    # print('Precision\t{0:.5f}'.format(precision * 100))
    # print('Recall\t{0:.5f}'.format(recall * 100))
    # print('F1\t{0:.5f}'.format(f1 * 100))
    # print('ExactMatch\t{0:.5f}'.format(accuracy * 100))

    with open(out_path, 'a') as fw:
        fw.write('Task [Total]: {0} {1}\n'.format(args.data, data_type))
        fw.write('Precision\tRecall\tF1\tExactMatch\n')
        fw.write('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}\n'.format(precision * 100, recall * 100, f1 * 100, accuracy * 100))

    return {'total_precision': precision, 'total_recall': recall, 'total_f1': f1, 'total_exact_match': accuracy}


def make_save_dir(out_dir):
    version = 0
    if os.path.exists(out_dir):
        dirs = glob.glob(out_dir + 'version_*/')
        if len(dirs) > 0:
            ids = [int(os.path.basename(os.path.dirname(dir)).replace('version_', '')) for dir in dirs]
            version = max(ids) + 1
    else:
        os.makedirs(out_dir)

    save_dir = out_dir + 'version_{0}/'.format(version)
    os.makedirs(save_dir)
    return save_dir


def encode_sentence(sent, pair):
    if pair == None:
        inputs = tokenizer(sent, padding=False, truncation=False, is_split_into_words=True, return_offsets_mapping=True,
                           return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda'),
                            inputs['token_type_ids'].to('cuda'))
    else:
        inputs = tokenizer(text=sent, text_pair=pair, padding=False, truncation=True,
                           is_split_into_words=True,
                           return_offsets_mapping=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda'),
                            inputs['token_type_ids'].to('cuda'))

    return outputs.hidden_states[args.layer][0], inputs['input_ids'][0], inputs['offset_mapping'][0]


def prepare_inputs(data_type):
    sents1, sents2, align_lists = load_WA_corpus(args.data, data_type, args.sure_and_possible)
    golds = [set(alignments.split()) for alignments in align_lists]

    hidden_outputs, input_ids, offset_mappings = [], [], []
    for s1, s2 in zip(tqdm(sents1), sents2):
        # print(' '.join(['{0}: {1} / '.format(i, w) for i, w in enumerate(s1)]))
        # print(' '.join(['{0}: {1} / '.format(i, w) for i, w in enumerate(s2)]))
        # print(alignments)
        if args.pair_encode:
            hidden_output, input_id, offset_map = encode_sentence(s1, s2)
            hidden_outputs.append(hidden_output)
            input_ids.append(input_id)
            offset_mappings.append(offset_map)
        else:
            hidden_output, input_id, offset_map = encode_sentence(s1, None)
            hidden_outputs.append(hidden_output)
            input_ids.append(input_id)
            offset_mappings.append(offset_map)

            hidden_output, input_id, offset_map = encode_sentence(s2, None)
            hidden_outputs.append(hidden_output)
            input_ids.append(input_id)
            offset_mappings.append(offset_map)

    if args.centering:  # Center BERT embedding: https://aclanthology.org/2020.eval4nlp-1.6/
        # Centering token embeddings
        mean_vec = torch.zeros(model.config.hidden_size, dtype=torch.float32, device='cuda')
        denominator = 0.0
        for idx in range(len(hidden_outputs)):
            mean_vec += torch.sum(hidden_outputs[idx], dim=0)
            denominator += hidden_outputs[idx].shape[0]
        mean_vec /= denominator

        for idx in range(len(hidden_outputs)):
            hidden_outputs[idx] = hidden_outputs[idx] - mean_vec

    s1_vecs, s2_vecs = [], []
    if args.pair_encode:
        for idx in range(len(hidden_outputs)):
            hidden_output = hidden_outputs[idx]
            input_id = input_ids[idx]
            offset_mapping = offset_mappings[idx]
            s1_vec, s2_vec = convert_to_word_embeddings(offset_mapping, input_id, hidden_output, tokenizer, True)

            # assert len(sents1[idx]) == s1_vec.shape[0]
            # assert len(sents2[idx]) == s2_vec.shape[0]

            s1_vecs.append(s1_vec)
            s2_vecs.append(s2_vec)
    else:
        for idx in range(int(len(hidden_outputs) / 2)):
            hidden_output = hidden_outputs[idx * 2]
            input_id = input_ids[idx * 2]
            offset_mapping = offset_mappings[idx * 2]
            vec = convert_to_word_embeddings(offset_mapping, input_id, hidden_output, tokenizer, False)
            # assert len(sents1[idx]) == vec.shape[0]
            s1_vecs.append(vec)

            hidden_output = hidden_outputs[idx * 2 + 1]
            input_id = input_ids[idx * 2 + 1]
            offset_mapping = offset_mappings[idx * 2 + 1]
            vec = convert_to_word_embeddings(offset_mapping, input_id, hidden_output, tokenizer, False)
            # assert len(sents2[idx]) == vec.shape[0]
            s2_vecs.append(vec)

    return s1_vecs, s2_vecs, sents1, sents2, golds


def final_evaluation(aligner, threshold, s1_vecs, s2_vecs, golds, sents1, sents2, data_type, final_result_path):
    aligner.compute_alignment_matrixes(s1_vecs, s2_vecs)
    predictions = aligner.get_alignments(threshold)
    log = evaluate(golds, predictions, data_type, final_result_path)
    log_null = eval_null_alignments(golds, sents1, sents2, predictions, data_type, final_result_path)
    log_total = evaluate_total_score(golds, predictions, sents1, sents2, data_type, final_result_path)

    predictions_with_cost = aligner.get_alignments(threshold, assign_cost=True)
    with open(os.path.dirname(final_result_path) + '/{0}_alignments_{1:.4f}_{1:.4f}.txt'.format(data_type, aligner.tau,
                                                                                                threshold),
              'w') as fw:
        fw.write('Sentence_1\tSentence_2\tGold\tPrediction\n')
        for s1, s2, gold_alignments, alignments in zip(sents1, sents2, golds, predictions_with_cost):
            fw.write('{0}\t{1}\t{2}\t{3}\n'.format(' '.join(s1), ' '.join(s2), ' '.join(gold_alignments),
                                                   ' '.join(alignments)))


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True).to('cuda').eval()
    distortion = load_distortion_setting('distortion_setting.yml', args.data, args.sure_and_possible)

    dev_s1_vecs, dev_s2_vecs, dev_sents1, dev_sents2, dev_golds = prepare_inputs('dev')
    test_s1_vecs, test_s2_vecs, test_sents1, test_sents2, test_golds = prepare_inputs('test')

    for ot_type in args.ot_type:
        out_dir = args.out + '{0}_sure-possible-{1}/{2}_{3}_{4}/{5}/'.format(args.data,
                                                                             args.sure_and_possible,
                                                                             ot_type,
                                                                             args.weight_type, args.dist_type,
                                                                             args.seed)
        out_dir = make_save_dir(out_dir)
        dev_log_path = out_dir + 'dev_log.txt'
        dev_final_result_path = out_dir + 'dev_report.txt'
        test_final_result_path = out_dir + 'test_report.txt'

        best_thresh = 0
        best_log = {'total_precision': 0.0, 'total_recall': 0.0, 'total_f1': 0.0, 'total_exact_match': 0.0}

        thresh_range = np.linspace(0.0, 1.0, num=100, endpoint=True)
        search_hypara_num = 50
        if ot_type == 'uot-mm':
            if args.div_type == 'kl':
                hypara_range = np.logspace(-4, 0, num=200, endpoint=True)[1:]
            else:
                hypara_range = np.logspace(-3, 3, num=200, endpoint=True)[1:]
        else:
            hypara_range = np.linspace(0.0, 1.0, num=search_hypara_num + 1, endpoint=False)[1:]
            patience = search_hypara_num

        if ot_type == 'uot' or (ot_type == 'pot' and args.sinkhorn):
            best_hypara = 0
            no_improve_cnt = 0
            for hypara in tqdm(hypara_range, desc=ot_type + ' Hyper-parameter search'):
                aligner = Aligner(ot_type, args.sinkhorn, args.chimera, args.dist_type, args.weight_type, distortion, 0,
                                  hypara, out_dir, div_type=args.div_type)
                aligner.compute_alignment_matrixes(dev_s1_vecs, dev_s2_vecs)
                improved = False
                for th in thresh_range:
                    predictions = aligner.get_alignments(th)
                    log = evaluate_total_score(dev_golds, predictions, dev_sents1, dev_sents2, 'dev', dev_log_path)
                    if log['total_f1'] > best_log['total_f1']:
                        best_hypara = hypara
                        best_thresh = th
                        best_log = log
                        improved = True
                    if log['total_f1'] == 0.0:
                        break
                if not improved:
                    no_improve_cnt += 1
                else:
                    no_improve_cnt = 0
                if no_improve_cnt >= patience:
                    break

            with open(dev_log_path, 'a') as fw:
                fw.write('*******************\n')
                fw.write('Best alignment Hyperparameter:\t{0:.5f}\t{1:.5f}\n'.format(best_hypara, best_thresh))

            aligner = Aligner(ot_type, args.sinkhorn, args.chimera, args.dist_type, args.weight_type, distortion,
                              best_thresh,
                              best_hypara, out_dir, div_type=args.div_type)
            final_evaluation(aligner, best_thresh, dev_s1_vecs, dev_s2_vecs, dev_golds, dev_sents1, dev_sents2,
                             'dev', dev_final_result_path)
            final_evaluation(aligner, best_thresh, test_s1_vecs, test_s2_vecs, test_golds, test_sents1, test_sents2,
                             'test', test_final_result_path)

        elif (ot_type == 'pot' and not args.sinkhorn) or ot_type == 'uot-mm':
            best_hypara = 0
            no_improve_cnt = 0
            for hypara in tqdm(hypara_range, desc=ot_type + ' Hyper-parameter search'):
                aligner = Aligner(ot_type, args.sinkhorn, args.chimera, args.dist_type, args.weight_type, distortion, 0,
                                  hypara,
                                  out_dir, div_type=args.div_type)
                aligner.compute_alignment_matrixes(dev_s1_vecs, dev_s2_vecs)
                improved = False
                predictions = aligner.get_alignments(0)
                log = evaluate_total_score(dev_golds, predictions, dev_sents1, dev_sents2, 'dev', dev_log_path)
                if log['total_f1'] > best_log['total_f1']:
                    best_hypara = hypara
                    best_log = log
                    improved = True
                if not improved:
                    no_improve_cnt += 1
                else:
                    no_improve_cnt = 0
                if no_improve_cnt >= patience:
                    break

            with open(dev_log_path, 'a') as fw:
                fw.write('*******************\n')
                fw.write('Best alignment Hyperparameter:\t{0:.5f}\n'.format(best_hypara))

            aligner = Aligner(ot_type, args.sinkhorn, args.chimera, args.dist_type, args.weight_type, distortion, 0,
                              best_hypara,
                              out_dir, div_type=args.div_type)
            final_evaluation(aligner, 0, dev_s1_vecs, dev_s2_vecs, dev_golds, dev_sents1, dev_sents2,
                             'dev', dev_final_result_path)
            final_evaluation(aligner, 0, test_s1_vecs, test_s2_vecs, test_golds, test_sents1, test_sents2,
                             'test', test_final_result_path)

        elif args.sinkhorn or ot_type == 'none':
            aligner = Aligner(ot_type, args.sinkhorn, args.chimera, args.dist_type, args.weight_type, distortion, 0, 0,
                              out_dir, div_type=args.div_type)
            aligner.compute_alignment_matrixes(dev_s1_vecs, dev_s2_vecs)

            for th in tqdm(thresh_range, desc=ot_type + ' Threshold search'):
                predictions = aligner.get_alignments(th)
                log = evaluate_total_score(dev_golds, predictions, dev_sents1, dev_sents2, 'dev', dev_log_path)
                if log['total_f1'] > best_log['total_f1']:
                    best_thresh = th
                    best_log = log
                if log['total_f1'] == 0.0:
                    break

            with open(dev_log_path, 'a') as fw:
                fw.write('*******************\n')
                fw.write('Best alignment Hyperparameter:\t{0:.5f}\n'.format(best_thresh))

            aligner.thresh = best_thresh
            aligner.save_hyper_parameters()
            final_evaluation(aligner, best_thresh, dev_s1_vecs, dev_s2_vecs, dev_golds, dev_sents1, dev_sents2,
                             'dev', dev_final_result_path)
            final_evaluation(aligner, best_thresh, test_s1_vecs, test_s2_vecs, test_golds, test_sents1, test_sents2,
                             'test', test_final_result_path)
        else:
            aligner = Aligner(ot_type, args.sinkhorn, args.chimera, args.dist_type, args.weight_type, distortion, 0, 0,
                              out_dir,
                              div_type=args.div_type)
            final_evaluation(aligner, 0, dev_s1_vecs, dev_s2_vecs, dev_golds, dev_sents1, dev_sents2, 'dev',
                             dev_final_result_path)
            final_evaluation(aligner, 0, test_s1_vecs, test_s2_vecs, test_golds, test_sents1, test_sents2,
                             'test', test_final_result_path)
