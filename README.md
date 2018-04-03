# PRPN

### Parsing Reading Predict Network

This repository contains the code used for word-level language model and unsupervised parsing experiments in [Neural Language Modeling by Jointly Learning Syntax and Lexicon](https://openreview.net/forum?id=rkgOLb-0W) paper, originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).
If you use this code or our results in your research, we'd appreciate if you cite our apper as following:

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

## Steps

1. Install PyTorch 0.2 and NLTK

2. Download PTB data. Note that the two tasks, i.e., language modeling and unsupervised parsing share the same model strucutre but require different formats of the PTB data. For language modeling we need the standard 10,000 word [Penn Treebank corpus](https://github.com/pytorch/examples/tree/75e435f98ab7aaa7f82632d4e633e8e03070e8ac/word_language_model/data/penn) data, and for parsing we need [Penn Treebank Parsed](https://catalog.ldc.upenn.edu/ldc99t42) data.

3. Scripts and commands

  	+  Language Modeling
  	```python main_LM.py --cuda --tied --hard --data /path/to/your/data```

    The default setting in `main_LM.py` achieves a test perplexity of approximately `60.97` on PTB test set.

  	+ Unsupervised Parsing
    ```python main_UP.py --cuda --tied --hard```

    The default setting in `main_UP.py` achieves an unlabeled f1 of approximately `0.70` on the standard test set of PTB WSJ10 subset. For visualizing the parsed sentence trees in nested bracket form, and evaluate the trained model, please run
	```test_phrase_grammar.py```
