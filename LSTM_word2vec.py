import os
import gensim
from gensim.models import Word2Vec


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = os.getcwd() + '/trainfile.txt'
        for line in open(corpus_path):
            yield line.split()


sentences = MyCorpus()
w2v_model = gensim.models.Word2Vec(sentences=sentences, min_count=1, vector_size=100)

w2v_model.save("word2vec.model")
