# PRPN

### Parsing Reading Predict Network

This repository contains the code used for word-level language model experiments in [Neural Language Modeling by Jointly Learning Syntax and Lexicon](https://openreview.net/forum?id=rkgOLb-0W) paper, originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).

+ Install PyTorch 0.2
+ Download PTB data
+ Run `python main.py --cuda --tied --hard`
+ `demo.py` provide unlabeled parsing result and visualization of attention and gate for input sentences. 

The default setting trains a PTB model achieves test perplexities of approximately `61.27`.
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

Python 2.7 and PyTorch 0.2 are required for the current codebase.
