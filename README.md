# OTAlign: Optimal Transport based Monolingual Word Alignment
<img src="./assets/uot_norm_cos_2.svg" alt="Alignment example by OTAlign" width="50%">

# Prerequisite
- See `src/requirements.txt`
- Please collect word alignment datasets: MultiMWA, Edinburgh++, MSR-RTE
  - Place them in a `data/` directory
  - Preprocessing codes for Edinburgh++ and MSR-RTE are in `src/preprocess`

# Unsupervised Word Alignment
For details, please refer to the arguments in `src/unsupervised_alignment.py`
``` shell
UN_OUTDIR=../out/unsupervised/
SEED=42
DATA=mtref
OT=uot
WT=uniform
DT=cos
$ python unsupervised_alignment.py --data $DATA --sure_and_possible --model bert-base-uncased --centering --pair_encode --layer -3 --out $UN_OUTDIR --ot_type $OT --weight_type $WT --dist_type $DT --seed $SEED
```
# Supervised Word Alignment
For details, please refer to the arguments in `src/supervised_alignment.py`

**Note**
Supervised word alignment uses hyperparameters estimated in the unsupervised setting. You first need to run unsupervised word alignment. 


``` shell
SU_OUTDIR=../out/supervised/
BATCH=64
PATIENCE=5

$ python python supervised_alignment.py --batch $BATCH --out $SU_OUTDIR --data $DATA --sure_and_possible --model bert-base-uncased --ot_type $OT --weight_type $WT --dist_type $DT --seed $SEED --patience $PATIENCE --unsupervised_dir $UN_OUTDIR
```

# Citation
Please cite our ACL2023 paper if you use this repository:
> Yuki Arase, Han Bao, and Sho Yokoi. 2023. Unbalanced Optimal Transport for Unbalanced Word Alignment. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3966–3986, Toronto, Canada. Association for Computational Linguistics.

``` shell
@inproceedings{arase-etal-2023-unbalanced,
    title = "Unbalanced Optimal Transport for Unbalanced Word Alignment",
    author = "Arase, Yuki  and
      Bao, Han  and
      Yokoi, Sho",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.219",
    doi = "10.18653/v1/2023.acl-long.219",
    pages = "3966--3986",
    abstract = "Monolingual word alignment is crucial to model semantic interactions between sentences.In particular, null alignment, a phenomenon in which words have no corresponding counterparts, is pervasive and critical in handling semantically divergent sentences. Identification of null alignment is useful on its own to reason about the semantic similarity of sentences by indicating there exists information inequality. To achieve unbalanced word alignment that values both alignment and null alignment, this study shows that the family of optimal transport (OT), i.e., balanced, partial, and unbalanced OT, are natural and powerful approaches even without tailor-made techniques.Our extensive experiments covering unsupervised and supervised settings indicate that our generic OT-based alignment methods are competitive against the state-of-the-arts specially designed for word alignment, remarkably on challenging datasets with high null alignment frequencies.",
}
``` 

# Contact 
If you have any questions about codes in this repository, please contact Yuki Arase via email or simply post an issue :speech_balloon:

