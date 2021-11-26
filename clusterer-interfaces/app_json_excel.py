import os
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request
from flask_restplus import Api, Resource, fields
from utils import remote_clustering, response_func

from app_csv_csv import csv_name_space as csv_ns
from app_excel_excel import excel_name_space as excel_ns

load_dotenv(".env")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

CORS(app, resources={r"/api/*": {"enabled": "true",
                                 "headers": [
                                     "Authorization",
                                     "Origin",
                                     "Access-Control-Request-Method",
                                     "Access-Control-Request-Headers",
                                     "Accept",
                                     "Accept-Charset",
                                     "Accept-Encoding",
                                     "Accept-Language",
                                     "Cache-Control",
                                     "Content-Type",
                                     "Cookie",
                                     "DNT",
                                     "Pragma",
                                     "Referer",
                                     "User-Agent",
                                     "X-Forwarded-For"
                                 ]
                                 }}, supports_credentials=True)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает *.xlsx')
input_data = name_space.model("Insert JSON",
                              {"texts": fields.List(fields.String(description="Insert texts", required=True)),
                               "score": fields.Float(description="Distance", required=True)}, )

# https://debugthis.dev/flask/2019-10-09-building-a-restful-api-using-flask-restplus/
name_space_csv = api.add_namespace(csv_ns)
name_space_excel = api.add_namespace(excel_ns)

CLUSTERING_URL = os.environ.get("CLUSTERING_URL")
if CLUSTERING_URL is None: raise Exception('Env var CLUSTERING_URL not defined')


@name_space.route('/json_excel')
class ClusteringJsonExcel(Resource):
    @name_space.expect(input_data)
    def post(self):
        """POST method on input json file with texts and score, output clustering texts as xlsx file."""
        json_data = request.json
        clustering_texts_df = remote_clustering(json_data, CLUSTERING_URL, upload_type="json")
        return response_func(clustering_texts_df)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4500)
