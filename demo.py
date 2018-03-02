import argparse
import copy
import numpy
import torch
from torch.autograd import Variable
from hinton import plot

import matplotlib.pyplot as plt

import data

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max+1:]) > 0:
            tree2 = build_tree(depth[idx_max+1:], sen[idx_max+1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree

def MRG(tr):
    if isinstance(tr, str):
        return '(' + tr + ')'
        # return tr + ' '
    else:
        s = '('
        for subtr in tr:
            s += MRG(subtr)
        s += ')'
        return s

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
print model

model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

while True:
    sens = raw_input('Input a sentences:')
    words = sens.strip().split()
    x = numpy.array([corpus.dictionary[w] for w in words])
    input = Variable(torch.LongTensor(x[:, None]))

    hidden = model.init_hidden(1)
    _, hidden = model(input, hidden)

    gates = model.gates.squeeze().data.numpy()

    parse_tree = build_tree(gates, words)
    print MRG(parse_tree)
