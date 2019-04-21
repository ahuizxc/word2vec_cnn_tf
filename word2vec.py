from gensim.models import word2vec
import logging



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus("../input/text8") 

model = word2vec.Word2Vec(sentences, size=200) 
model.save("text8.model")

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec

model = KeyedVectors.load("text8.model.wv.vectors.npy")

print(model['word'])



