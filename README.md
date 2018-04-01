# PRPN

### Parsing Reading Predict Network

This repository contains the code used for word-level language model and unsupervised parsing experiments in [Neural Language Modeling by Jointly Learning Syntax and Lexicon](https://openreview.net/forum?id=rkgOLb-0W) paper, originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).

+ Install PyTorch 0.2 and NLTK
+ Download PTB data (Note that language model and unsupervised parsing task request the PTB data in different format)
+ Run `python main_LM.py --cuda --tied --hard --data /path/to/your/data` for language model task
+ Run `python main_UP.py --cuda --tied --hard` for unsupervised parsing task (a Tesla K80 is used for running this experiment)
+ `demo.py` generate unlabeled parsing tree for input sentences. 

The default setting in `main_LM.py` trains a PTB language model achieves test perplexities of approximately `60.97`.
The default setting in `main_UP.py` trains a PTB unsupervised parsing model achieves unlabeled f1 of approximately `0.70`.
If you use this code or our results in your research, please cite:

```
@inproceedings{
shen2018neural,
title={Neural Language Modeling by Jointly Learning Syntax and Lexicon},
author={Yikang Shen and Zhouhan Lin and Chin-wei Huang and Aaron Courville},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=rkgOLb-0W},
}
```

## Software Requirements

Python 2.7, NLTK and PyTorch 0.2 are required for the current codebase.
