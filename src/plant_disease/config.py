"""集中式运行时配置。

此模块定义 ``Settings`` 数据类与 ``load_settings()``，是整个应用**唯一**读取
``os.environ`` 的地方（外加根目录三个入口脚本里读 ``PLANT_DISEASE_DEBUG`` 来配
置日志级别）。

# 配置分层
- **代码里的默认值**：所有非敏感配置（HOST、PORT、WEIGHTS_PATH、CLASSES_TXT
  …）的默认值就在下面的 ``Settings`` / ``load_settings`` 里。仓库 clone 下来
  开箱即用，不需要 .env。
- **.env 文件**：仅放敏感信息（API key），且被 ``.gitignore`` 忽略不会进仓库。
  想临时覆盖某个非敏感配置（比如改端口）时，可以在自己机器的 .env 里加一行
  ``PORT=8000``，不会被 commit。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，"1/true/yes/on"（大小写不敏感）算 True，其余算 False。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


@dataclass(frozen=True)
class Settings:
    """应用运行期使用的全部配置。

    通过 ``load_settings()`` 构造，构造完之后是 ``frozen``（不可改），避免运行
    时被某段代码意外修改。

    Attributes:
        weights_path: MobileNetV2 权重 .pth 文件路径
        classes_txt: 61 类映射表路径（``resources/actual_classed_v2.txt``）
        llm_provider: ``mock`` / ``openai`` / ``baidu`` / ``alibaba``
        dashscope_api_key: 阿里通义 DashScope key
        openai_api_key: OpenAI key
        baidu_api_key: 百度文心 API Key
        baidu_secret_key: 百度文心 Secret Key
        flask_debug: 是否开 Flask debug（仅本地开发；生产留 False）
        host: Web 服务绑定地址（默认仅本机 127.0.0.1）
        port: Web 服务端口
    """

    weights_path: Path
    classes_txt: Path
    llm_provider: str
    dashscope_api_key: str = ""
    openai_api_key: str = ""
    baidu_api_key: str = ""
    baidu_secret_key: str = ""
    flask_debug: bool = False
    host: str = "127.0.0.1"
    port: int = 5000


def load_settings() -> Settings:
    """从环境变量构造 ``Settings``；缺失的项一律用默认值，不会报错。

    每个字段的环境变量名见函数体；想新增配置项请同步更新 ``Settings`` 字段、
    本函数读取语句、以及 ``.env.example``（如该项需要在 .env 里覆盖）。
    """
    return Settings(
        weights_path=Path(os.environ.get("WEIGHTS_PATH", "resources/mobilenetv2_best.pth")),
        classes_txt=Path(os.environ.get("CLASSES_TXT", "resources/actual_classed_v2.txt")),
        llm_provider=os.environ.get("LLM_PROVIDER", "mock").strip().lower(),
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        baidu_api_key=os.environ.get("BAIDU_API_KEY", ""),
        baidu_secret_key=os.environ.get("BAIDU_SECRET_KEY", ""),
        flask_debug=_bool_env("FLASK_DEBUG", False),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
    )
