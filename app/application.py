"""Flask application factory and the page (non API) routes."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, send_from_directory

from app.api import BLUEPRINTS
from app.database.db import init_database
from app.feature_engineering.extractor import FEATURE_GROUPS, FEATURE_NAMES
from app.services import model_service, prediction_service, report_service
from config.config import FIGURE_DIR, FlaskConfig
from config.logging_config import get_logger

LOGGER = get_logger(__name__)


def create_app(config_object=FlaskConfig) -> Flask:
    """Build and configure the Flask application."""
    application = Flask(__name__)
    application.config.from_object(config_object)

    for blueprint in BLUEPRINTS:
        application.register_blueprint(blueprint)

    try:
        init_database()
    except Exception as error:  # pragma: no cover - the app still serves pages
        LOGGER.warning("Database initialisation failed: %s", error)

    register_pages(application)
    register_error_handlers(application)
    return application


def register_pages(application: Flask) -> None:
    @application.route("/")
    def index():
        return render_template(
            "index.html",
            models=model_service.list_models(),
            ready=model_service.is_ready(),
            feature_count=len(FEATURE_NAMES),
        )

    @application.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html", data=report_service.dashboard_payload())

    @application.route("/models")
    def models():
        return render_template(
            "models.html",
            models=model_service.list_models(),
            comparison=model_service.comparison_table(),
        )

    @application.route("/analysis")
    def analysis():
        return render_template(
            "analysis.html",
            importance=report_service.feature_importance_report(),
            experiments=report_service.experiment_report(),
            feature_groups={name: len(items) for name, items in FEATURE_GROUPS.items()},
            feature_count=len(FEATURE_NAMES),
        )

    @application.route("/predictions")
    def predictions():
        return render_template(
            "prediction.html",
            history=prediction_service.history(limit=100),
            summary=prediction_service.summary(),
        )

    @application.route("/reports")
    def reports():
        return render_template(
            "reports.html",
            report=report_service.model_comparison_report(),
            experiments=report_service.experiment_report(),
        )

    @application.route("/report-figures/<path:filename>")
    def report_figures(filename):
        return send_from_directory(FIGURE_DIR, filename)

    @application.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "models_ready": model_service.is_ready(),
                "feature_count": len(FEATURE_NAMES),
            }
        )


def register_error_handlers(application: Flask) -> None:
    @application.errorhandler(404)
    def not_found(error):  # pragma: no cover - trivial
        return jsonify({"error": "Not found"}), 404

    @application.errorhandler(500)
    def server_error(error):  # pragma: no cover - trivial
        LOGGER.exception("Unhandled server error")
        return jsonify({"error": "Internal server error"}), 500
