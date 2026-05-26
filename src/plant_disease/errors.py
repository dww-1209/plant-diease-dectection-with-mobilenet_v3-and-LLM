"""自定义异常层级。

整个项目的所有异常都继承自 ``PlantDiseaseError``。Web 路由层根据具体子类把
异常翻译成对应的 HTTP 状态码：

- ``InferenceError`` → 500 / 503（服务器内部错或模型未加载）
- ``LLMServiceError`` → 502（上游 LLM 调用失败，属于 Bad Gateway 语义）
- ``LLMConfigError`` → 400（用户未配置 API key 等，属于 Bad Request）
"""


class PlantDiseaseError(Exception):
    """项目所有自定义异常的根基类，便于一次性 ``except`` 兜底。"""


class InferenceError(PlantDiseaseError):
    """推理过程出错。

    触发场景：
    - 权重文件不存在 / 加载失败
    - 上传的图片无法解码（不是合法图片 / 文件损坏）
    - 模型前向传播抛异常（极少见，通常是显存不足 / 输入张量形状不对）
    """


class LLMServiceError(PlantDiseaseError):
    """LLM 上游调用失败。

    触发场景：网络超时、连接错误、HTTP 4xx/5xx、响应不是合法 JSON、响应缺
    少预期字段等。**不**包含本地配置错误（用 ``LLMConfigError``）。
    """


class LLMConfigError(LLMServiceError):
    """LLM 提供商所需的配置缺失或无效（如 API key 没填、凭证被拒绝）。

    继承自 ``LLMServiceError`` 是为了让"想 except 所有 LLM 类错误"的代码自然生效，
    但路由层会优先 except 这个更具体的类，给用户返回 400 而不是 502。
    """
