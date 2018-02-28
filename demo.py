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


# def build_tree_depth(depth, sen):
#     depth = depth.tolist()
#     sorted_idx = numpy.argsort(depth)
#     parse_tree = copy.copy(sen)
#     i2i = numpy.arange(len(parse_tree))
#     for idx in sorted_idx:
#         idx_mapped = i2i[idx]
#         new_node = parse_tree[idx_mapped]
#         d = depth[idx_mapped]
#         if idx < len(sen) - 1 and depth[idx_mapped + 1] <= d:
#             new_node = [new_node, parse_tree.pop(idx_mapped + 1)]
#             depth.pop(idx_mapped + 1)
#             i2i[idx + 1:] -= 1
#         if idx > 0 and depth[idx_mapped - 1] < d:
#             idx_mapped -= 1
#             new_node = [parse_tree.pop(idx_mapped), new_node]
#             depth.pop(idx_mapped)
#             i2i[idx:] -= 1
#         parse_tree[idx_mapped] = new_node
#     return parse_tree

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
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

while True:
    sens = raw_input('Input a sentences:')
    hidden = model.init_hidden(1)
    for s in sens.split('\t'):
        words = s.strip().split()
        x = numpy.array([corpus.dictionary[w] for w in words])
        input = Variable(torch.LongTensor(x[:, None]))
        
        # hidden = model.init_hidden(1)
        output, hidden = model(input, hidden)
        output = output.squeeze().data.numpy()[:-1]
        output = numpy.log(softmax(output))
        output = numpy.pad(output, ((1, 0), (0, 0)), 'constant', constant_values=0)
        output = numpy.exp(-output[range(len(words)), x])

        attentions = model.attentions.squeeze().data.numpy()
        gates = model.gates.squeeze().data.numpy()
        phrase = []
        sentence = []
        for i in range(len(words)):
            print '%15s\t%7.1f\t%.3f\t%s' % (words[i], output[i], gates[i], plot(attentions[i], 1).replace('\n', '\t'))
            midx = numpy.argmax(gates[i])
            if midx > 0:
                if phrase != []:
                    sentence.append(phrase)
                phrase = []
            phrase.append(words[i])
        sentence.append(phrase)

        print output[1:].mean()

        parse_tree = build_tree(gates, words)
        print MRG(parse_tree)
        # parse_tree = build_tree_depth(gates, words)
        # print MRG(parse_tree)
