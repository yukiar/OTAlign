import glob, codecs, re, random


def load_MSR_RTE_corpus(file_path, sure_and_possible):
    sent1_list = []
    sent2_list = []
    alignment_list = []
    p = re.compile(r'\({([\sp\d]+?)/ / }\)?')

    with codecs.open(file_path, encoding='utf-8') as f:
        lines = [l.strip().replace('\ufeff', '') for l in f.readlines()] # Remove BOF char that could not removed by nkf
        for lid in range(len(lines)):
            if lid % 3 == 0:
                sent1 = lines[lid + 1]
                s1_words = sent1.split()

                # convert alignment
                sent2_alignments = lines[lid + 2]
                a_lists = p.findall(sent2_alignments)
                s2_words = p.sub(r'', sent2_alignments).split()
                assert len(a_lists) == len(s2_words)
                alignments = []
                for tid, alist in enumerate(a_lists[1:]):  # 0 is NULL
                    for align in alist.strip().split():
                        if align[0] == 'p':
                            sid = int(align[1:]) - 1  # 1-base indexing -> 0-base
                            assert sid < len(s1_words) and sid >= 0
                            if sure_and_possible:
                                alignments.append('{0}-{1}'.format(sid, tid))
                            else:
                                pass
                        else:
                            sid = int(align) - 1  # 1-base indexing -> 0-base
                            assert sid < len(s1_words) and sid >= 0
                            alignments.append('{0}-{1}'.format(sid, tid))

                sent1_list.append(sent1)
                sent2_list.append(' '.join(s2_words[1:]))  # 0 is NULL
                alignment_list.append(set(alignments))

    return (sent1_list, sent2_list, alignment_list)


def save_file(out_path, sents1, sents2, alignments):
    with open(out_path, 'w') as fw:
        for s1, s2, alignment in zip(sents1, sents2, alignments):
            fw.write('{0}\t{1}\t{2}\n'.format(s1, s2, ' '.join(list(alignment))))


def select_from_list(input_list, indices):
    new_list = [input_list[i] for i in indices]
    return new_list


if __name__ == '__main__':
    # corpus_dir = '../data/RTE-2006-Aligned/Test/'
    # sure_and_possible = True
    # out_path = '../data/RTE-2006-Aligned/sure-and-possible_'
    corpus_dir = '../../data/RTE-2006-Aligned/Test/'
    sure_and_possible = False
    out_path = '../../data/RTE-2006-Aligned/sure_'

    all_annotators = []
    for file_path in glob.glob(corpus_dir + '*.align.txt'):
        examples = load_MSR_RTE_corpus(file_path, sure_and_possible)
        all_annotators.append(examples)

    # Consolidate annotations
    sents1, sents2, alignments = [], [], []
    for id in range(len(all_annotators[0][0])):
        sents1.append(all_annotators[0][0][id])
        sents2.append(all_annotators[0][1][id])
        # Take annotations for which at least 2 annotators agreed
        alignment_ab = all_annotators[0][2][id] & all_annotators[1][2][id]
        alignment_ac = all_annotators[0][2][id] & all_annotators[2][2][id]
        alignment_bc = all_annotators[1][2][id] & all_annotators[2][2][id]
        alignment = set(alignment_ab | alignment_bc | alignment_ac)
        alignments.append(alignment)

    if 'Dev' in corpus_dir:
        # Split to 600 train and 200 dev
        random.seed(42)
        idx_list = list(range(len(sents1)))
        random.shuffle(idx_list)
        train_ids = idx_list[:600]
        dev_ids = idx_list[600:]
        save_file(out_path + 'train.txt', select_from_list(sents1, train_ids), select_from_list(sents2, train_ids),
                  select_from_list(alignments, train_ids))
        save_file(out_path + 'dev.txt', select_from_list(sents1, dev_ids), select_from_list(sents2, dev_ids),
                  select_from_list(alignments, dev_ids))
    else:
        save_file(out_path + 'test.txt', sents1, sents2, alignments)

    print('Done!')
