"""Flask app factory."""

from __future__ import annotations

import logging

from flask import Flask

from plant_disease.config import Settings
from plant_disease.errors import InferenceError
from plant_disease.model import InferenceModel
from plant_disease.web.routes import register_routes

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> Flask:
    app = Flask(
        __name__,
        template_folder="../../../templates",
        static_folder="../../../static",
    )
    app.config["SETTINGS"] = settings
    app.config["INIT_ERROR"] = None
    app.config["INFERENCE_MODEL"] = None

    try:
        app.config["INFERENCE_MODEL"] = InferenceModel(
            weights_path=settings.weights_path,
            classes_txt=settings.classes_txt,
        )
    except InferenceError as exc:
        logger.warning("inference model failed to initialize: %s", exc)
        app.config["INIT_ERROR"] = str(exc)

    register_routes(app)
    return app
