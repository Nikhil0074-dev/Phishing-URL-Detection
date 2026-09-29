"""URL validation and feature extraction endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.feature_engineering.url_parser import InvalidURLError
from app.services import feature_service, url_service

url_bp = Blueprint("url", __name__, url_prefix="/api")


def _payload_url() -> str:
    data = request.get_json(silent=True) or {}
    return str(data.get("url") or request.args.get("url") or "")


@url_bp.route("/validate", methods=["POST", "GET"])
def validate():
    result = url_service.validate_url(_payload_url())
    return jsonify(result), (200 if result["valid"] else 400)


@url_bp.route("/features", methods=["POST"])
def features():
    url = _payload_url()
    validation = url_service.validate_url(url)
    if not validation["valid"]:
        return jsonify({"error": validation["error"]}), 400
    try:
        return jsonify(feature_service.features_for_url(url)), 200
    except InvalidURLError as error:
        return jsonify({"error": str(error)}), 400


@url_bp.route("/features/batch", methods=["POST"])
def features_batch():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls")
    if not isinstance(urls, list) or not urls:
        return jsonify({"error": "Provide a non-empty 'urls' list."}), 400
    if len(urls) > 500:
        return jsonify({"error": "At most 500 URLs per request."}), 400
    records = feature_service.features_for_many(urls)
    return jsonify({"count": len(records), "results": records}), 200
