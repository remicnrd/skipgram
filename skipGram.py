import sys
import math
import nltk
import argparse
import numpy as np
import pandas as pdk

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

def get_all_words(sentences):
    allWords = list()
    for sentence in sentences:
        for word in sentence:
            word = word.lower()
            if len(word) > 1 and word.isalnum():
                allWords.append(word)
    return allWords

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    def __init__(self, sentences, minCount):
        self.wordList = list()  # a list of word element, with the word and its frequency
        self.wordHash = dict()  # a dictionnary of word and their hash value (place in the text)
        self.build(sentences, minCount)

    def build(self, sentences, minCount):
        wordList = list()
        wordHash = dict()      
        for sentence in sentences:
            for word in sentence:
                if word not in wordList:
                    wordHash[word] = len(wordList)  # The length of the list is used as our hash/counter as well
                    wordList.append(Word(word))     # We append the Word element 
                wordList[wordHash[word]].count += 1 # and keep track of frequency for mincount filtering later          
        
        # Create a new list without rare words
        cleanList = list()
        cleanList.append(Word('{unknownWord}'))
        for word in wordList:
            if word.count < minCount:
                cleanList[0].count += word.count
            else:
                cleanList.append(word)
        cleanList.sort(key=lambda word : word.count, reverse=True)
        # Change our wordHash accordingly
        wordHash = dict()
        for i, w in enumerate(cleanList):
            wordHash[w.word] = i
            
        self.wordList = cleanList
        self.wordHash = wordHash
    
    def getHash(self, wordList):
        return [self.wordHash[word] if word in self.wordList else self.wordHash['{unknownWord}'] for word in wordList]



class UnigramTable:
    def __init__(self, vocabulary):
        sumOfWeights = sum([math.pow(w.count, 3/4) for w in vocabulary.wordList])
        tableSize = int(1e8)
        table = np.zeros(tableSize, dtype=np.uint32)
        p = 0
        i = 0
        for j, word in enumerate(vocabulary.wordList):
            p += float(math.pow(word.count, 3/4))/sumOfWeights
            while i < tableSize and float(i) / tableSize < p:
                table[i] = j
                i += 1
        self.table = table

    def negativeSample(self, count):
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
        vocabulary = Vocabulary(sentences, self.minCount)  # init vocab from train file
        unigramTable = UnigramTable(vocabulary)       # init table from train file
        allWords = get_all_words(self.sentences)

        # 2: we create the NN and learn the weight matrix
        for window in [self.winSize]: 
            for dim in [self.nEmbed]: 
                layer0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(len(vocabulary.wordList), dim))
                layer1 = np.zeros(shape=(len(vocabulary.wordList), dim))
                tokens = vocabulary.getHash(allWords)
                for token_idx, token in enumerate(tokens):
                    current_window = np.random.randint(low=1, high=window+1)
                    context_start = max(token_idx - current_window, 0)
                    context_end = min(token_idx + current_window + 1, len(tokens))
                    context = tokens[context_start:token_idx] + tokens[token_idx+1:context_end]
                    for context_word in context:
                        ne = np.zeros(dim)
                        classifiers = [(token, 1)] + [(target, 0) for target in unigramTable.negativeSample(self.negativeRate)]
                        for target, label in classifiers:
                            z = np.dot(layer0[context_word], layer1[target])
                            p = sigmoid(z)
                            g = stepsize * (label - p)
                            ne += g * layer1[target]
                            layer1[target] += g * layer0[context_word] 
                        layer0[context_word] += ne

    def save(self,path):
        # 3: we save the weight matrix to the specified path
        with open(path, "wb") as f:
            pickle.dump((vocabulary,layer0), f)

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        vec1 = layer0[vocabulary.getHash([word1])[0]]
        vec2 = layer0[vocabulary.getHash([word2])[0]]
        dot_product = np.dot(vec1, vec2)
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        return dot_product / (n1 * n2)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            a,b = pickle.load(f)  
        return a,b

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
