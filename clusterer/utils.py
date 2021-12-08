import os
import numpy as np
import pandas as pd
from tfidf_tools import key_words_tfidf_get
from sklearn.metrics.pairwise import cosine_similarity
from transformers_tools import clustering_texts_func


def tree_traversal(tx_tree):
    """Representation ClustersTree as dictionary."""
    d = {}
    for item in tx_tree.items:
        if isinstance(item, TextsCluster):
            d[item.root] = item.items
        else:
            d = {**d, **tree_traversal(item)}
    return {tx_tree.root: d}


def dict2tuples(d, keys=()):
    """Convert dictionary to list of tuples"""
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            result.extend(dict2tuples(v, keys + (k,)))
        else:
            result += [keys + (k, x) for x in v]
    return result


class ClustersTree:
    """Object for several clusters representation"""

    def __init__(self, root: str, items: []):
        self.root = root
        self.items = items

    def root_hash(self):
        """Cluster center hash"""
        return hash(self.root)

    def dict_format(self):
        return tree_traversal(ClustersTree(self.root, self.items))

    def tuples_format(self):
        return dict2tuples(self.dict_format())

    def pandas_format(self):
        """Representation ClustersTree as pandas DataFrame."""
        return pd.DataFrame(self.tuples_format())


class TextsCluster(ClustersTree):
    """Class for texts clusters, items must be list of str."""

    def __init__(self, root: str, items: []):
        super().__init__(root, items)
        for item in items:
            assert isinstance(item, str), "Not true values in items! All values must be str!"

    def texts_hashes(self):
        """Cluster texts hashes"""
        return [hash(tx) for tx in self.items]


def clusters_link(leading_clusters_list: [TextsCluster], slave_clusters_list: []) -> []:
    """Function concatenates list of TextsCluster objects with list of TextsCluster or
     ClusterTree objects. Returns list of TextsCluster or ClusterTree objects (trees constructor)"""
    assert len(slave_clusters_list) == sum([len(x.items) for x in leading_clusters_list]), \
        "sum of texts in leading_clusters_list must be equal number of TextsClusters in slave cluster list"

    def pair_clusters_link(leading_cluster: TextsCluster, slave_clusters: []):
        """Function connections leading cluster with slave clusters by texts and roots hashes respectively.
        Texts of leading_cluster must be the same as roots of slave clusters.
        """
        leading_hashes_texts = list(zip(leading_cluster.texts_hashes(), leading_cluster.items))
        leading_hashes_sorted, leading_texts_sorted = zip(*sorted(leading_hashes_texts, key=lambda x: x[0]))
        slave_hashes_items = [(tc.root_hash(), tc.items) for tc in slave_clusters]
        slave_hashes_sorted, slave_items_sorted = zip(*sorted(slave_hashes_items, key=lambda x: x[0]))

        slave_items_list = []
        for x, y in zip(leading_texts_sorted, slave_items_sorted):
            if sum([isinstance(i, str) for i in y]):
                slave_items_list.append(TextsCluster(x, y))
            else:
                slave_items_list.append(ClustersTree(x, y))
        return ClustersTree(leading_cluster.root, slave_items_list)

    link_clusters = []
    for lead_tc in leading_clusters_list:
        temp_clusters = [slave_tc for slave_tc in slave_clusters_list if
                         slave_tc.root_hash() in lead_tc.texts_hashes()]
        link_clusters.append((lead_tc, temp_clusters))

    return [pair_clusters_link(x, y) for x, y in link_clusters]


def clusters_centroids(texts_matrix: [()]) -> [TextsCluster]:
    """Returns list of TextsCluster objects where root is a centroid of items (texts) in each."""

    def get_centroid(texts: [], vectors_matrix: np.array):
        """Returns the centroid of an array of texts"""

        def cluster_name_number(vectors: np.array) -> np.array:
            """Function get vectors, finds vector most close to average of vectors and returns it's number."""
            # weight_average_vector = np.average(vectors, axis=0, weights=vectors)
            weight_average_vector = np.average(vectors, axis=0)
            weight_average_vector_ = weight_average_vector.reshape(1, weight_average_vector.shape[0])
            similarity_to_average = cosine_similarity(vectors, weight_average_vector_)
            return np.argmax(similarity_to_average)

        num = cluster_name_number(vectors_matrix)
        return texts[num]

    return [TextsCluster(get_centroid(texts, matrix), texts) for texts, matrix in texts_matrix]


def cluster_key_words(texts, kw_n=3) -> [TextsCluster]:
    """"""
    key_words = key_words_tfidf_get([" ".join(tx) for tx in texts], n=kw_n)
    return [TextsCluster(rt, txs) for rt, txs in zip(key_words, texts)]


def tree3levels_construction(tree_root: str, texts: [str], score: float) -> [()]:
    """"""
    clustering_texts_matrix1 = clustering_texts_func(score, texts)
    text_clusters_level1 = clusters_centroids(clustering_texts_matrix1)
    clustering_texts = clustering_texts_func(score, [x.root for x in text_clusters_level1], texts_and_vectors=False)
    text_clusters_level2 = cluster_key_words(clustering_texts)
    clusters_tree = clusters_link(text_clusters_level2, text_clusters_level1)
    text_clusters_level3 = [TextsCluster(tree_root, [x.root for x in text_clusters_level2])]
    return clusters_link(text_clusters_level3, clusters_tree)[0].tuples_format()


if __name__ == "__main__":
    df = pd.read_csv(os.path.join("data", "clastering_test_data.csv"), header=None)
    texts = list(df[0])

    r = tree3levels_construction("книжка", texts)
    print(r)



