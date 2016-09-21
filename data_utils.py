import os, re, math
import numpy as np
import cPickle as pickle

def load_task(data_dir):
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    train_file = [f for f in files if 'train' in f][0]
    valid_file = [f for f in files if 'trial' in f][0]
    test_file = [f for f in files if 'test' in f][0]
    train_data = get_sick_data(train_file)
    valid_data = get_sick_data(valid_file)
    test_data = get_sick_data(test_file)
    return train_data, valid_data, test_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def get_sick_data(file_name):
    class_dict = {"entailment":0, "neutral":1, "contradiction":2}
    with open(file_name) as f:
        haed = f.readline()
        data = []
        for line in f:
            line = str.lower(line.strip())
            nid, sent1, sent2, score, label = line.split('\t')
            sent1 = tokenize(sent1)
            sent2 = tokenize(sent2)
            score = float(score)
            label = class_dict[label]
            data.append((sent1, sent2, score, label))
        return data


def vectorize_data(data, word_idx, sentence_size):
    S1, S2 = [], []
    LABEL = []
    SCORE = []
    s1_length, s2_length = [], []
    for sent1, sent2, _score, _label in data:
        # pad to memory_size
        l_s1 = max(0, sentence_size - len(sent1))
        s1 = [word_idx[w] for w in sent1] + [0] * l_s1

        l_s2 = max(0, sentence_size - len(sent2))
        s2 = [word_idx[w] for w in sent2] + [0] * l_s2

        label = np.zeros(3)
        label[_label] = 1

        score = np.zeros(5)
        floor = math.floor(_score)
        if floor != 5:
            score[floor] = _score - floor
        score[floor - 1] = floor + 1 - _score
        assert _score == np.dot(range(1, 6), score)

        S1.append(s1)
        S2.append(s2)
        LABEL.append(label)
        SCORE.append(score)
        s1_length.append(len(sent1))
        s2_length.append(len(sent2))
    return np.array(S1), np.array(S2), np.array(LABEL), np.array(SCORE), \
                np.array(s1_length).astype(np.int32), np.array(s2_length).astype(np.int32)


def load_embedding(word_idx, Index=True):
    file_name = 'data/vocab.pkl'

    if Index:
        return pickle.load(open(file_name, 'rb'))

    def load_google_vec(fname):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
        return word_vecs
    model = load_google_vec('/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin')
    #model = Word2Vec.load_word2vec_format('/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
    embedding = []
    for c in word_idx:
        if c in model:
            embedding.append(model[c])
        else:
            embedding.append(np.random.uniform(0.1, 0.1, 300))
    embedding = np.array(embedding, dtype='float32')
    pickle.dump(embedding, open(file_name, 'wb'))
    return embedding
