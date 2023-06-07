import glob, codecs, re, random, json


def load_annotation(annotation, sure_and_possible, sent1, sent2):
    s1_words = sent1.split()
    s2_words = sent2.split()

    alignments = []
    for align in annotation['S']:
        sid = align[0]
        assert sid >= 0 and sid < len(s1_words)

        for tid in align[1]:
            assert tid >= 0 and tid < len(s2_words)
            alignments.append('{0}-{1}'.format(sid, tid))

    if sure_and_possible:
        for align in annotation['P']:
            sid = align[0]
            assert sid >= 0 and sid < len(s1_words)

            for tid in align[1]:
                assert tid >= 0 and tid < len(s2_words)
                alignments.append('{0}-{1}'.format(sid, tid))

    return alignments


def load_Edinburgh_corpus(file_path, sure_and_possible):
    sent1_list = []
    sent2_list = []
    alignment_list = []

    with open(file_path) as f:
        df = json.load(f)

    for item in df['paraphrases']:
        sent1 = item['S']['string']
        sent2 = item['T']['string']
        align_A, align_C = [], []
        if 'A' in item['annotations']:
            align_A = load_annotation(item['annotations']['A'], sure_and_possible, sent1, sent2)
            align_A = set(align_A)
            alignment = align_A
        if 'C' in item['annotations']:
            align_C = load_annotation(item['annotations']['C'], sure_and_possible, sent1, sent2)
            align_C = set(align_C)
            alignment = align_C
        if len(align_A) > 0 and len(align_C) > 0:
            alignment = align_A & align_C

        sent1_list.append(sent1)
        sent2_list.append(sent2)
        alignment_list.append(alignment)

    return (sent1_list, sent2_list, alignment_list)


def save_file(out_path, sents1, sents2, alignments):
    with open(out_path, 'w') as fw:
        for s1, s2, alignment in zip(sents1, sents2, alignments):
            fw.write('{0}\t{1}\t{2}\n'.format(s1, s2, ' '.join(list(alignment))))


def select_from_list(input_list, indices):
    new_list = [input_list[i] for i in indices]
    return new_list


if __name__ == '__main__':
    corpus_path = '../../data/edinburgh/test.json'
    sure_and_possible = False
    out_path = '../data/edinburgh/sure_'

    # Consolidate annotations
    sents1, sents2, alignments = load_Edinburgh_corpus(corpus_path, sure_and_possible)

    if 'train' in corpus_path:
        # Split to 514 train and 200 dev
        random.seed(42)
        idx_list = list(range(len(sents1)))
        random.shuffle(idx_list)
        train_ids = idx_list[:514]
        dev_ids = idx_list[514:]
        save_file(out_path + 'train.txt', select_from_list(sents1, train_ids), select_from_list(sents2, train_ids),
                  select_from_list(alignments, train_ids))
        save_file(out_path + 'dev.txt', select_from_list(sents1, dev_ids), select_from_list(sents2, dev_ids),
                  select_from_list(alignments, dev_ids))
    else:
        save_file(out_path + 'test.txt', sents1, sents2, alignments)

    print('Done!')
