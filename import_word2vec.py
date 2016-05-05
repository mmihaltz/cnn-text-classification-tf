#! /usr/bin/env python
"""
Run this once to export a numpy matrix[i] = vector for word index i
for the training data vocabulary using word2vec pre-trained embeddings
+ random vectors for unknown words
into a pickle file.
"""

import data_helpers
import numpy as np
import cPickle

# full path and name of word2vec embedding binary file to use
word2vec_bin_file = "../word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin"

# output pickle file name
word2vec_pickle_file = "word2vec-vocab.pickle"


def load_bin_vec(fname, vocab):
    """
    from: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    Loads 300x1 word vecs from Google (Mikolov) word2vec for entries in vocab
    :returns {word: wordvec}
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, k=300):
    """
    based on: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    For words in vocab but not in word_vecs, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones (from word2vec)
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def get_W(word_vecs, vocab_inv, k=300):
    """
    source: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    Get word embedding matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    W = np.zeros(shape=(vocab_size, k), dtype='float32')
    for i in range(vocab_size):
        W[i] = word_vecs[vocab_inv[i]]
    return W


if __name__ == '__main__':
    print("Loading vocabulary...")
    _x, _y, vocabulary, vocabulary_inv = data_helpers.load_data()
    print('Vocabulary size: ' + str(len(vocabulary)))
    print('vocabulary sample:')
    for k in list(sorted(vocabulary.keys()))[:25]:
        print(k, vocabulary[k])
    print('Loading word2vec embeddings...')
    w2v = load_bin_vec(word2vec_bin_file, vocabulary)
    print('Words found in word2vec: {}'.format(len(w2v)))
    add_unknown_words(w2v, vocabulary, k=300)
    print('Size of w2v after adding vectors for unknown words: {}'.format(len(w2v)))
    print('w2v sample:')
    for k in list(sorted(w2v.keys()))[:25]:
        print(k, w2v[k][:10])
    W = get_W(w2v, vocabulary_inv)
    print('Matrix {}: {}'.format(type(W), len(W)))
    cPickle.dump(W, open(word2vec_pickle_file, 'wb'))
    print('Dumped to file: {}'.format(word2vec_pickle_file))