# BaM: An Enhanced Training Scheme for Balanced and Comprehensive Multi-interest Learning

This project is a pytorch implementation of 'BaM: An Enhanced Training Scheme for Balanced and Comprehensive Multi-interest Learning'.
BaM (<U/>Ba</U>lanced Interest Learning for <U/>M</U>ulti-interest Recommendation) is an effective and generally applicable training scheme for balanced learning of multi-interest and it achieves up to 15.01% higher accuracy in sequential recommendation compared to the best competitor, resulting in the state-of-the-art performance.
This project provides executable source code with adjustable arguments and preprocessed datasets used in the paper.

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)

## Usage

There are 3 folders and each consists of:
- data: preprocessed datasets
- runs: pre-trained models for each dataset
- src: source codes

You can run a demo script 'demo.sh' to compare the performance of BaM on Movies & TV dataset by evaluating pre-trained model.
The result looks as follows:
```
users: 22747
items: 17848
interactions: 841607
model loaded from ./runs/movies/bam/
test: 22747it [00:15, 1474.88it/s]
test recall[@10, @20]: [0.0819, 0.1153], test nDCG[@10, @20]: [0.0475, 0.0559]
```

You can also train the model by running 'main.py'.
There are 4 arguments you can control:
- path (any string, default is 'run1'): the path to save the trained model and training log.
- dataset ('movies' or 'books')
- model ('mind' or 'comirec'): the backbone model to use.
    * 'mind': MIND from "Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19). Association for Computing Machinery, New York, NY, USA, 2615–2623. https://doi.org/10.1145/3357384.3357814".
    * 'comirec': ComiRec-SA from "Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Controllable Multi-Interest Framework for Recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 2942–2951. https://doi.org/10.1145/3394486.3403344".
- selection ('hard' or 'bam'): the method of selecting interest from multi-interests.
    * 'hard': hard selection from previous methods.
    * 'bam': The proposed soft-selection method BaM.

For example, you can train the model for Books dataset with BaM on ComiRec at 'bam' by following code:
```
python src/main.py --path bam --dataset books --model comirec --selection bam
```


You can evaluate the trained_model by running 'main.py' with the argument 'test' as True:
```
python src/main.py --path bam --dataset books --model comirec --selection bam --test True
```

## Datasets
Preprocessed data are included in the data directory.
| Dataset | Users | Items | Interactions | Density |
| --- | ---: | ---: | ---: | ---: |
|Movies & TV (movies)| 22,747 | 17,848 | 841,607 | 0.20% |
|Books (books) | 14,905 | 13,642 | 626,702 | 0.31% |

The original datasets are available at https://amazon-reviews-2023.github.io.
