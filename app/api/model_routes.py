"""Model metadata endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify

from app.services import model_service

model_bp = Blueprint("model", __name__, url_prefix="/api/models")


@model_bp.route("", methods=["GET"])
@model_bp.route("/", methods=["GET"])
def list_models():
    return jsonify({"models": model_service.list_models()}), 200


@model_bp.route("/comparison", methods=["GET"])
def comparison():
    table = model_service.comparison_table()
    if not table:
        return jsonify(
            {"error": "No metrics found. Run 'python -m app.ml.train' first."}
        ), 503
    return jsonify({"comparison": table, "best_model": model_service.best_model()}), 200


@model_bp.route("/status", methods=["GET"])
def status():
    return jsonify(
        {
            "ready": model_service.is_ready(),
            "models": model_service.list_models(),
        }
    ), 200
