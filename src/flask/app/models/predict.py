from flask_restx import fields
from ..instance import server

request = server.api.model("Predict Request", {
    "id_sap": fields.Integer(required=True, description="ID SAP"),
    "pf_pj": fields.String(required=True, description="PF/PJ"),
    "contratado_ofertas_simples": fields.Integer(required=True, description="Contratado Ofertas Simples"),
    "utilizado_ofertas_simples": fields.Integer(required=True, description="Utilizado Ofertas Simples"),
    "leads_form": fields.Integer(required=True, description="Leads Form"),
    "equipe": fields.String(required=True, description="Equipe"),
    "utilizado_destaque": fields.Integer(required=True, description="Utilizado Destaque"),
    "valor_mensal": fields.Float(required=True, description="Valor Mensal"),
    "quantidade_mes": fields.Integer(required=True, description="Somatório dos Meses os Quais o Cliente Possui Vínculo com o Serviço"),
    "status_pagamento": fields.String(required=True, description="Quality Score do Cliente"),
    "regiao": fields.String(required=True, description="Regiao"),
    "oficina": fields.String(required=True, description="Oficina"),
    "tipo_de_plano": fields.String(required=True, description="Tipo de Plano"),
    "frequencia_de_faturamento": fields.String(required=True, description="Frequencia de Faturamento"),
})

response = server.api.model("Predict Response", {
    "id_sap": fields.Integer(required=True, description="ID SAP"),
    "churn": fields.Boolean(required=True, description="Churn")
})