from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

class API(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('img', type=int)
        img_id = parser.parse_args()["img"]

        return f"got id {img_id}"


api.add_resource(API, '/api', endpoint='api')

app.run(host="0.0.0.0", port=5000)

