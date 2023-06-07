import torch, re, yaml, os
import numpy as np
import collections.abc as container_abcs
from ot.backend import get_backend


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, s1_sents, s2_sents, input_s2t, alignment_list, subword_to_word_map, s2_start_idxes):
        self.s1_sents = [' '.join(sent) for sent in s1_sents]
        self.s2_sents = [' '.join(sent) for sent in s2_sents]
        self.input_s2t = input_s2t
        self.align_strs = alignment_list
        self.align_tuples = []
        for aligns in alignment_list:
            tuples = []
            for pair in aligns.split():
                i_j = pair.split('-')
                tuples.append((int(i_j[0]), int(i_j[1])))
            self.align_tuples.append(tuples)
        self.subword_to_word_maps = subword_to_word_map
        self.s2_start_idxes = s2_start_idxes

    def __getitem__(self, idx):
        item = {'s2t_' + key: val[idx].clone().detach() for key, val in self.input_s2t.items()}
        item['s1_sents'] = self.s1_sents[idx]
        item['s2_sents'] = self.s2_sents[idx]
        item['alignments'] = self.align_strs[idx]
        item['atuples'] = self.align_tuples[idx]
        item['subword_to_word_map'] = self.subword_to_word_maps[idx]
        item['s2_start_idx'] = self.s2_start_idxes[idx]
        return item

    def __len__(self):
        return len(self.align_strs)


def alignment_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "alignment_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return alignment_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: alignment_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(alignment_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # Allow arbitrary length lists
        return batch
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def send_batch_to_gpu(elem, device):
    if isinstance(elem, torch.Tensor):
        return elem.to(device)
    elif isinstance(elem, dict):
        return {key: send_batch_to_gpu(val, device) for key, val in elem.items()}
    elif isinstance(elem, list):
        return [send_batch_to_gpu(val, device) for val in elem]
    else:
        return elem


def read_Word_Alignment_Dataset(filename, sure_and_possible=False, transpose=False):
    data = []
    for line in open(filename):
        ID, sent1, _, sent2, _, _, _, sure_align, poss_align, *useless = line.strip('\n').split('\t')
        my_dict = {}
        my_dict['source'] = sent1
        my_dict['target'] = sent2
        my_dict['sureAlign'] = sure_align
        my_dict['possibleAlign'] = poss_align
        data.append(my_dict)
    sent1_list = []
    sent2_list = []
    alignment_list = []
    for i in range(len(data)):
        if transpose:
            source = data[i]['target']
            target = data[i]['source']
        else:
            source = data[i]['source']
            target = data[i]['target']
        alignment = data[i]['sureAlign']
        if sure_and_possible:
            alignment += ' ' + data[i]['possibleAlign']
        my_label = []
        for item in alignment.split():  # reverse the alignment
            i, j = item.split('-')
            if transpose:
                my_label.append(str(j) + '-' + str(i))
            else:
                my_label.append(str(i) + '-' + str(j))
        alignment = ' '.join(my_label)
        sent1_list.append(source.lower().split())
        sent2_list.append(target.lower().split())
        alignment_list.append(alignment)
    return (sent1_list, sent2_list, alignment_list)


def read_Preprocessed_Word_Alignment_Dataset(file_path):
    sent1_list = []
    sent2_list = []
    alignment_list = []

    with open(file_path) as f:
        for line in f:
            array = [item.strip() for item in line.split('\t')]
            sent1_list.append(array[0].lower().split())
            sent2_list.append(array[1].lower().split())
            alignment_list.append(array[2])

    return (sent1_list, sent2_list, alignment_list)


def load_WA_corpus(data, type, sure_and_possible):
    if data in ['mtref', 'wiki', 'msr', 'edinburgh']:
        load_data = data
    else:
        if type == 'test':
            load_data = data
        else:  # Use mtref train & dev sets for newsela and arxiv
            load_data = 'mtref'

    if load_data == 'mtref':
        examples = read_Word_Alignment_Dataset('../data/MultiMWA-data/MultiMWA-MTRef/mtref-' + type + '.tsv',
                                               sure_and_possible, True)
    elif load_data == 'wiki':
        examples = read_Word_Alignment_Dataset('../data/MultiMWA-data/MultiMWA-Wiki/wiki-' + type + '.tsv',
                                               sure_and_possible, False)
    elif load_data == 'newsela':
        examples = read_Word_Alignment_Dataset('../data/MultiMWA-data/MultiMWA-Newsela/newsela-' + type + '.tsv',
                                               sure_and_possible, False)
    elif load_data == 'arxiv':
        examples = read_Word_Alignment_Dataset('../data/MultiMWA-data/MultiMWA-arXiv/arxiv-' + type + '.tsv',
                                               sure_and_possible, False)
    elif load_data == 'msr':
        data_path = '../data/RTE-2006-Aligned/'
        if sure_and_possible:
            data_path += 'sure-and-possible_' + type + '.txt'
        else:
            data_path += 'sure_' + type + '.txt'
        examples = read_Preprocessed_Word_Alignment_Dataset(data_path)
    elif load_data == 'edinburgh':
        data_path = '../data/edinburgh/'
        if sure_and_possible:
            data_path += 'sure-and-possible_' + type + '.txt'
        else:
            data_path += 'sure_' + type + '.txt'
        examples = read_Preprocessed_Word_Alignment_Dataset(data_path)

    return examples


def load_lr_setting(file_path, patience, sure_and_possible, ot_type):
    with open(file_path, 'r') as yml:
        config = yaml.safe_load(yml)
    sp_key = 'sure_and_possible' if sure_and_possible else 'sure'
    lr = float(config['patience_{0}'.format(patience)][sp_key][ot_type]['lr'])
    return lr


def load_distortion_setting(file_path, data, sure_and_possible):
    with open(file_path, 'r') as yml:
        config = yaml.safe_load(yml)

    sure_type = 'sure_and_possible' if sure_and_possible else 'sure'
    distortion = config[sure_type][data]['distortion']

    return distortion


def load_ot_setting(dir_path, ot_type):
    if ot_type == 'ot':
        return 0.0
    elif ot_type == 'pot':
        files = os.listdir(dir_path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(dir_path, f))]
        assert len(files_dir) == 2
        # Load setting without Sinkhorn
        with open(os.path.join(dir_path, files_dir[0], 'dev_log.txt')) as f:
            dev_logs = f.readlines()
        settings = dev_logs[-1].strip().split('\t')
        hyp = float(settings[1])
        return hyp
    else:
        files = os.listdir(dir_path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(dir_path, f))]
        assert len(files_dir) == 1
        with open(os.path.join(dir_path, files_dir[0], 'dev_log.txt')) as f:
            dev_logs = f.readlines()
        settings = dev_logs[-1].strip().split('\t')
        hyp = float(settings[1])
        thresh = float(settings[2])

        return hyp

    raise Exception('OT setting load error!')


def get_token_word_mapping(all_input_s2t, tokenizer):
    all_subword_to_word_map, all_s2_start_idx = [], []
    for token_ids, offset_mapping in zip(all_input_s2t['input_ids'], all_input_s2t['offset_mapping']):
        word_idx = -1
        subword_to_word_conv = np.full((len(token_ids)), -1)
        # Bug in hugging face tokenizer? Sometimes Metaspace is inserted
        metaspace = getattr(tokenizer.decoder, "replacement", None)
        metaspace = tokenizer.decoder.prefix if metaspace is None else metaspace
        tokenizer_bug_idxes = [i for i, x in enumerate(tokenizer.convert_ids_to_tokens(token_ids)) if x == metaspace]

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

        sep_tok_indices = [i for i, x in enumerate(token_ids) if x == tokenizer.sep_token_id]
        s2_start_idx = subword_to_word_conv[
            sep_tok_indices[0] + np.argmax(subword_to_word_conv[sep_tok_indices[0]:] > -1)]

        all_subword_to_word_map.append(subword_to_word_conv)
        all_s2_start_idx.append(s2_start_idx)

    return all_subword_to_word_map, all_s2_start_idx


def get_aligned_indices(alignments):
    aligned_s1_indices, aligned_s2_indices = set(), set()
    for a in alignments:
        i_j = a.split('-')
        aligned_s1_indices.add(int(i_j[0]))
        aligned_s2_indices.add(int(i_j[1]))
    return aligned_s1_indices, aligned_s2_indices


def grow_diag_final(e2f, f2e, len_e, len_f):
    alignments = grow_diag(e2f, f2e, len_e, len_f)
    alignments = final(len_e, len_f, e2f | f2e, alignments)
    return alignments


def get_aligned_points(alignments):
    aligned_e = set()
    aligned_f = set()
    for a in alignments:
        a = a.split('-')
        aligned_e.add(int(a[0]))
        aligned_f.add(int(a[1]))
    return aligned_e, aligned_f


def get_neighbors(e_idx, f_idx, e_len, f_len):
    neighboring = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    new_points = [(e_idx + e, f_idx + f) for e, f in neighboring if
                  e_idx + e >= 0 and e_idx + e < e_len and f_idx + f >= 0 and f_idx + f < f_len]

    return new_points


def grow_diag(e2f, f2e, len_e, len_f):
    alignments = e2f & f2e
    union = e2f | f2e
    aligned_e, aligned_f = get_aligned_points(alignments)
    new_points_added = True

    while new_points_added:
        new_points_added = False
        for e_idx in range(len_e):
            for f_idx in range(len_f):
                if '{0}-{1}'.format(e_idx, f_idx) in alignments:
                    for e_new, f_new in get_neighbors(e_idx, f_idx, len_e, len_f):
                        new_point = '{0}-{1}'.format(e_new, f_new)
                        if (e_new not in aligned_e or f_new not in aligned_f) and new_point in union:
                            alignments.add(new_point)
                            aligned_e.add(e_new)
                            aligned_f.add(f_new)
                            new_points_added = True

    return alignments


def final(len_e, len_f, union, alignments):
    aligned_e, aligned_f = get_aligned_points(alignments)
    for e_new in range(len_e):
        for f_new in range(len_f):
            new_point = '{0}-{1}'.format(e_new, f_new)
            if (e_new not in aligned_e or f_new not in aligned_f) and new_point in union:
                alignments.add(new_point)
                aligned_e.add(e_new)
                aligned_f.add(f_new)

    return alignments


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def cosine_dist_matrix(matX, matY, eps=1e-08):
    num = torch.matmul(matX, matY.t())
    denom = torch.maximum(torch.full_like(num, eps),
                          torch.matmul(torch.norm(matX, dim=1, keepdim=True),
                                       torch.norm(matY, dim=1, keepdim=True).t()))
    cos_dist = 1 - num / denom
    return cos_dist


def convert_to_word_embeddings(offset_mapping, token_ids, hidden_tensors, tokenizer, pair):
    word_idx = -1
    subword_to_word_conv = np.full((hidden_tensors.shape[0]), -1)
    # Bug in hugging face tokenizer? Sometimes Metaspace is inserted
    metaspace = getattr(tokenizer.decoder, "replacement", None)
    metaspace = tokenizer.decoder.prefix if metaspace is None else metaspace
    tokenizer_bug_idxes = [i for i, x in enumerate(tokenizer.convert_ids_to_tokens(token_ids)) if
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
        ([torch.mean(hidden_tensors[subword_to_word_conv == word_idx], dim=0) for word_idx in range(word_idx + 1)]))

    if pair:
        sep_tok_indices = [i for i, x in enumerate(token_ids) if x == tokenizer.sep_token_id]
        s2_start_idx = subword_to_word_conv[
            sep_tok_indices[0] + np.argmax(subword_to_word_conv[sep_tok_indices[0]:] > -1)]

        s1_word_embeddigs = word_embeddings[0:s2_start_idx, :]
        s2_word_embeddigs = word_embeddings[s2_start_idx:, :]

        return s1_word_embeddigs, s2_word_embeddigs
    else:
        return word_embeddings


def split_list_with_window(l, slide, window_size):
    if len(l) <= window_size - slide:
        return [l]
    else:
        return [l[i:i + window_size] for i in range(0, len(l) - window_size + slide, slide)]


def min_max_scaling(C):
    eps = 1e-10
    # Min-max scaling for stabilization
    nx = get_backend(C)
    C_min = nx.min(C)
    C_max = nx.max(C)
    C = (C - C_min + eps) / (C_max - C_min + eps)
    return C


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        # kwargs['average_attn_weights'] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap
