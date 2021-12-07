# https://radimrehurek.com/gensim/models/tfidfmodel.html
import os
import pandas as pd
from gensim.models import TfidfModel
from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
from texts_processors import SimpleTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity


def key_words_tfidf_get(texts: [], n: int) -> [str]:
    """Function get list of texts, finds key words with tf-idf algorithm
    (word's vector should be close to text's vector) for each text and returns n of them."""

    tokenizer = SimpleTokenizerFast({})
    lm_texts = tokenizer(texts)
    dct = Dictionary(lm_texts)  # fit dictionary
    corpus = [dct.doc2bow(lm_tx) for lm_tx in lm_texts]  # convert corpus to BoW format
    model = TfidfModel(corpus)  # fit model

    texts_key_words = []
    for tx_num, lm_text in enumerate(lm_texts):
        lm_text_unique = set(lm_text)
        words_corps = [dct.doc2bow([w]) for w in lm_text_unique]
        words_vectors = [model[x] for x in words_corps]
        words_vectors_sparse = corpus2csc(words_vectors, len(dct))

        tx_tfidf_vec = model[corpus[tx_num]]
        tx_matrix_sparse = corpus2csc([tx_tfidf_vec], len(dct))
        cos_dist = cosine_similarity(words_vectors_sparse.T, tx_matrix_sparse.T)
        dist_words = list(zip(cos_dist, lm_text_unique))
        dist_words_sorted = sorted(dist_words, key=lambda x: x[0], reverse=True)
        texts_key_words.append(" ".join([x[1] for x in dist_words_sorted[:n]]))
    return texts_key_words


if __name__ == "__main__":
    df = pd.read_csv(os.path.join("data", "clastering_test_data.csv"), header=None)
    texts = list(df[0])

    r = key_words_tfidf_get(texts, n=3)
    print(r)
    print(len(r))
    print(t2)
