import os
from flask import Flask
from dotenv import load_dotenv
from flask_restplus import Api, Resource
from utils import remote_clustering, response_func, api_configurator

load_dotenv(".env")
CLUSTERING_URL = os.environ.get("CLUSTERING_URL")
if CLUSTERING_URL is None: raise Exception('Env var CLUSTERING_URL not defined')

app = Flask(__name__)
api = Api(app)

excel_name_space = api.namespace('api', 'загрузка файла excel с текстами (обязательно поле ""texts) и score (число)')
upload_parser = api_configurator(excel_name_space)


@excel_name_space.route('/excel_excel')
@excel_name_space.expect(upload_parser)
class ClusteringEcxelExcel(Resource):
    def post(self):
        """POST method on input xlsx file with texts and score, output clustering texts  as xlsx file."""
        args = upload_parser.parse_args()
        clustering_texts_df = remote_clustering(args, CLUSTERING_URL)
        return response_func(clustering_texts_df)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006)
