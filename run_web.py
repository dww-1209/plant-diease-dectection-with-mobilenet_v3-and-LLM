"""Web 服务入口。

【运行方式】

    python run_web.py

打开浏览器访问 http://localhost:5000

【环境变量】
    HOST, PORT, FLASK_DEBUG, LLM_PROVIDER, DASHSCOPE_API_KEY, ...
    复制 .env.example 为 .env 并按需填写即可。
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from plant_disease.config import load_settings
from plant_disease.web.app import create_app


def main() -> int:
    load_dotenv()
    level = (
        logging.DEBUG
        if os.environ.get("PLANT_DISEASE_DEBUG", "").lower() in {"1", "true"}
        else logging.INFO
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = load_settings()
    app = create_app(settings)
    # Flask 自带的 dev server，生产请用 gunicorn / uvicorn 等
    app.run(host=settings.host, port=settings.port, debug=settings.flask_debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
