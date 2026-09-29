"""Prediction endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.feature_engineering.url_parser import InvalidURLError
from app.ml.predict import ModelNotTrainedError
from app.services import prediction_service, url_service

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


def _request_data() -> dict:
    return request.get_json(silent=True) or {}


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    data = _request_data()
    url = str(data.get("url") or "")
    validation = url_service.validate_url(url)
    if not validation["valid"]:
        return jsonify({"error": validation["error"]}), 400

    try:
        result = prediction_service.analyse_url(
            url,
            model_name=data.get("model"),
            store=bool(data.get("store", True)),
            include_features=bool(data.get("include_features", False)),
        )
    except KeyError as error:
        return jsonify({"error": str(error)}), 400
    except ModelNotTrainedError as error:
        return jsonify({"error": str(error)}), 503
    except InvalidURLError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result), 200


@prediction_bp.route("/predict/compare", methods=["POST"])
def predict_compare():
    data = _request_data()
    url = str(data.get("url") or "")
    validation = url_service.validate_url(url)
    if not validation["valid"]:
        return jsonify({"error": validation["error"]}), 400
    try:
        result = prediction_service.compare_models_on_url(
            url, store=bool(data.get("store", False))
        )
    except ModelNotTrainedError as error:
        return jsonify({"error": str(error)}), 503
    return jsonify(result), 200


@prediction_bp.route("/predictions", methods=["GET"])
def predictions():
    limit = min(int(request.args.get("limit", 50)), 500)
    offset = max(int(request.args.get("offset", 0)), 0)
    return jsonify(
        {
            "count": limit,
            "offset": offset,
            "summary": prediction_service.summary(),
            "results": prediction_service.history(limit=limit, offset=offset),
        }
    ), 200
