from model.lstm import *
import gensim
from gensim.models import word2vec

sls = lstm("bestsem.p", load=True, training=False)

sa = "A truly wise man"
sb = "He is smart"



print sls.predict_similarity(sa, sb) * 4.0 + 1.0
