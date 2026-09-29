"""Report endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.services import report_service

report_bp = Blueprint("report", __name__, url_prefix="/api/reports")


@report_bp.route("/model-comparison", methods=["GET"])
def model_comparison():
    report = report_service.model_comparison_report()
    return jsonify(report), (200 if report["available"] else 503)


@report_bp.route("/feature-importance", methods=["GET"])
def feature_importance():
    report = report_service.feature_importance_report(request.args.get("model"))
    return jsonify(report), (200 if report.get("available") else 503)


@report_bp.route("/experiments", methods=["GET"])
def experiments():
    report = report_service.experiment_report()
    return jsonify(report), (200 if report.get("available") else 503)
