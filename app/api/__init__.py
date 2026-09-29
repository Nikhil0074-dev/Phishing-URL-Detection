"""REST API blueprints."""

from app.api.model_routes import model_bp  # noqa: F401
from app.api.prediction_routes import prediction_bp  # noqa: F401
from app.api.report_routes import report_bp  # noqa: F401
from app.api.url_routes import url_bp  # noqa: F401

BLUEPRINTS = (url_bp, prediction_bp, model_bp, report_bp)
