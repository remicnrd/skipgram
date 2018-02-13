from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalizene

import nltk


__authors__ = ['rémi canard','maria bosch vidal']
__emails__  = ['B00713672@essec.edu','B00720030@essec.edu']

def text2sentences(path):
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(nltk.word_tokenize(l))
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    def __init__(self, sentences, minCount):
        self.words = []     # a list of word element, with the word and its frequency
        self.word_map = {}  # a dictionnary of word and their hash value (place in the text)

        self.build_words(sentences, minCount)
        self.filter_for_rare_and_common()

    def build_words(self, sentences, minCount):
        words = []
        word_map = {}

        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in word_map:
                    word_map[word] = len(words)
                    words.append(Word(word))
                words[word_map[word]].count += 1
                i += 1
                
        self.words = words
        self.word_map = word_map

    def indices(self, tokens):
        return [self.word_map[token] if token in self else self.word_map['{rare}'] for token in tokens]

    def filter_for_rare_and_common(self):
        # Remove rare words and sort
        tmp = []
        tmp.append(Word('{rare}'))
        unk_hash = 0

        count_unk = 0
        for token in self.words:
            if token.count < minCount:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update word_map
        word_map = {}
        for i, token in enumerate(tmp):
            word_map[token.word] = i

        self.words = tmp
        self.word_map = word_map
        pass



class TableForNegativeSamples:
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab.words]) # Normalizing constants

        table_size = 1e8
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0 # Cumulative probability
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]



class mSkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount

    def train(self,stepsize, epochs):

        # 1: we receive a text and create the dictionnary and the Negative sample table
        vocab = Vocabulary(sentences, minCount)  # init vocab from train file
        table = TableForNegativeSamples(vocab) # init table from train file


        # 2: we create the NN and learn the weight matrix
        for window in [5]:             # Max window length
            for dim in [100]:          # Dimensionality of word embeddings
                
                # Initialize network
                nn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(len(vocab), dim))      # init first layer with random weights from a uniform distribution on the interval [-0.5, 0.5]/dim
                nn1 = np.zeros(shape=(len(vocab), dim))                                          # init second layer with 0

                alpha =  0.01                                                                    # Learning rate
                tokens = vocab.word_map.values()

                for token_idx, token in enumerate(tokens):
                    current_window = np.random.randint(low=1, high=window+1)                     # Randomize window size, where win is the max window size
                    context_start = max(token_idx - current_window, 0)
                    context_end = min(token_idx + current_window + 1, len(tokens))
                    context = tokens[context_start:token_idx] + tokens[token_idx+1:context_end]  # Turn into an iterator?

                    for context_word in context:
                        neu1e = np.zeros(dim)                                                    # Init neu1e with zeros
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(k_negative_sampling)]
                        for target, label in classifiers:
                            z = np.dot(nn0[context_word], nn1[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * nn1[target]              # Error to backpropagate to nn0
                            nn1[target] += g * nn0[context_word]  # Update nn1

                        # Update nn0
                        nn0[context_word] += neu1e

                # Save model to file
                save(vocab, nn0, 'output-%s-%d-%d-%d' % (input_filename, window, dim, word_phrase_passes))



    def save(self,path):
        # 3: we save the weight matrix to the specified path

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        raise NotImplementedError('implement it!')

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print sg.similarity(a,b)
