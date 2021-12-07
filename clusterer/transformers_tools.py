import os
import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer


def clustering_texts_func(score: float, texts: [], texts_and_vectors=True) -> []:
    """Function clusters texts """

    def grouped_func(data: list, texts_and_vectors: bool) -> [()]:
        """Function groups input list of data with format: [(label, vector, text)]
        into list of tuples,
        (texts: list of texts correspond to label,
        vectors_matrix: numpy matrix of vectors correspond to label)
        """
        data = sorted(data, key=lambda x: x[0])
        grouped_data = []
        for key, group_items in groupby(data, key=itemgetter(0)):
            temp_vectors, texts = zip(*[(x[1], x[2]) for x in group_items])
            vectors_matrix = np.vstack(temp_vectors)
            grouped_data.append((texts, vectors_matrix))
        if texts_and_vectors:
            return grouped_data
        else:
            texts_list, matrix_list = zip(*grouped_data)
        return texts_list

    vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=score,
                                        memory=os.path.join("cache"))

    vectors = vectorizer.encode([x.lower() for x in texts])

    clusters = clusterer.fit(vectors)
    lbs_vs_txs = [(lb, v, tx) for lb, v, tx in zip(clusters.labels_, vectors, texts)]
    return grouped_func(lbs_vs_txs, texts_and_vectors)
