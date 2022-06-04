from ..controllers.predict import Predicter
from flask import Flask
from flask_restx import Api, Resource
from ..instance import server
from ..models.predict import request, response

app, api = server.app, server.api

@api.route('/predict')
class Predict(Resource):

    @api.expect([request], validate=True)
    @api.marshal_list_with(response)
    def post(self):
        predicter = Predicter(api.payload)
        return predicter.run()

@api.route('/upload/batch/carteira-total')
class UploadFullCarteiraTotal(Resource):
    @api.expect([request], validate=True)
    @api.marshal_list_with(response)
    def post(self):
        predicter = Predicter(api.payload)
        return predicter.upload_full_carteira_total()

@api.route('/upload/batch/quality-score')
class UploadFullCarteiraParcelada(Resource):
    @api.expect([request], validate=True)
    @api.marshal_list_with(response)
    def post(self):
        predicter = Predicter(api.payload)
        return predicter.upload_full_carteira_parcelada()

@api.route('/upload/micro-batch/carteira-total')
class UploadFullCarteiraTotal(Resource):
    @api.expect([request], validate=True)
    @api.marshal_list_with(response)
    def post(self):
        predicter = Predicter(api.payload)
        return predicter.upload_full_carteira_total()

@api.route('/upload/micro-batch/quality-score')
class UploadFullCarteiraParcelada(Resource):
    @api.expect([request], validate=True)
    @api.marshal_list_with(response)
    def post(self):
        predicter = Predicter(api.payload)
        return predicter.upload_full_carteira_parcelada()
