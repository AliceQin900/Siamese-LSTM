# Siamese-LSTM

## pre-requirement
1. Download the word2vec model from https://code.google.com/archive/p/word2vec/ and download the file: GoogleNews-vectors-negative300.bin.gz
2. install theano, gensim

## Files
1. semtrain.p - training data (SICK2014 tarin+trail) format: (sent1, sent2, score, lable, paid_id)
2. semtest.p - testing date (SICK2014 test)
3. stsallrmf.p - all STS data.

4. dwords.p - dict, like remove ing, etc.
5. synsem - dict, wordnet synsets, format: ('key', ['answer', 'keys'])

## Scripts

Scripts: (in examples folder)
1. example.py : Load trained model to predict sentence similarity on a scale of 1.0-5.0
2. main.py : train the model, Load trained model and check Pearson, Spearman and MSE.
3. preprocess.py: load data, preprocess
4. model/lstm.py: Siamese-LSTM

## Refrence
Mueller, J and Thyagarajan, A.  Siamese Recurrent Architectures for Learning Sentence Similarity.  Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016).
 http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195