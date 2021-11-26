import os
from random import shuffle
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает JSON')
input_data = name_space.model("Insert JSON",
                              {"texts": fields.List(fields.String(description="Insert texts", required=True)),
                               "score": fields.Float(description="Distance", required=True)}, )


def clustering_func(vectorizer: SentenceTransformer, clusterer: AgglomerativeClustering, texts: []) -> {}:
    """Function for text collection clustering"""
    vectors = vectorizer.encode([x.lower() for x in texts])
    clusters = clusterer.fit(vectors)
    return {"texts_with_labels": list(zip([int(x) for x in clusters.labels_], texts))}


@name_space.route('/clusterer')
class Clustering(Resource):
    @name_space.expect(input_data)
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts as JSON file."""
        json_data = request.json
        texts_list = json_data["texts"]

        """restricting number of texts fragments (resource limit)"""
        shuffle(texts_list)
        clustering_texts = texts_list[:30000]
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=json_data['score'],
                                            memory=os.path.join("cache"))

        vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        return jsonify(clustering_func(vectorizer, clusterer, clustering_texts))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
