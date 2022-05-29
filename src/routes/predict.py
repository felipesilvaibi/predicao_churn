from src.controllers.predict import Predicter
from flask import Flask
from flask_restx import Api, Resource
from src.server.instance import server
from src.models.predict import request, response

app, api = server.app, server.api

@api.route('/predict')
class Predict(Resource):

    @api.expect([request], validate=True)
    @api.marshal_list_with(response)
    def post(self):
        predicter = Predicter(api.payload)
        return predicter.run()
