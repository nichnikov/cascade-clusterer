from random import shuffle
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields
from utils import tree3levels_construction


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает JSON')
input_data = name_space.model("Insert JSON",
                              {
                               "tree_name": fields.String(description="Insert texts", required=True),
                               "texts": fields.List(fields.String(description="Insert texts", required=True)),
                               "score": fields.Float(description="Distance", required=True)}, )



@name_space.route('/clusterer')
class Clustering(Resource):
    @name_space.expect(input_data)
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts as JSON file."""
        json_data = request.json
        texts_list = json_data["texts"]
        tree_root = json_data["tree_name"]
        score = json_data["score"]

        """restricting number of texts fragments (resource limit)"""
        shuffle(texts_list)
        clustering_texts = texts_list[:30000]
        resulting_tuples = tree3levels_construction(tree_root, clustering_texts, score)
        return jsonify({"texts_tree": resulting_tuples})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6005)
