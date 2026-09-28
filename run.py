"""Entry point for the Phishing URL Detection web application.

Usage::

    python run.py
    FLASK_PORT=8000 FLASK_DEBUG=1 python run.py
"""

from __future__ import annotations

from app.application import create_app
from config.config import FlaskConfig, ensure_directories
from config.logging_config import configure_logging

configure_logging()
ensure_directories()

app = create_app()

if __name__ == "__main__":
    app.run(host=FlaskConfig.HOST, port=FlaskConfig.PORT, debug=FlaskConfig.DEBUG)
