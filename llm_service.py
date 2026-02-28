"""
大模型服务模块 - 用于获取植物病害治理建议
支持多种大模型API：OpenAI、百度文心、阿里通义等
"""
import os
from typing import Optional
import requests


class LLMService:
    """大模型服务基类"""

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        # 优先使用DASHSCOPE_API_KEY（官方推荐），其次使用LLM_API_KEY
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("LLM_API_KEY", "")
        self.api_base = api_base

    def get_treatment_advice(
        self, plant_class: str, disease_name: str, disease_degree: str, health_status: str
    ) -> str:
        """
        获取植物病害治理建议
        
        Args:
            plant_class: 植物名称
            disease_name: 病害名称
            disease_degree: 患病程度
            health_status: 健康状况
            
        Returns:
            治理建议文本
        """
        raise NotImplementedError("子类必须实现此方法")


class OpenAIService(LLMService):
    """OpenAI API服务（支持GPT-3.5/4等）"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key)
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.model = model

    def get_treatment_advice(
        self, plant_class: str, disease_name: str, disease_degree: str, health_status: str
    ) -> str:
        """使用OpenAI API获取治理建议"""
        if not self.api_key:
            return "错误：未配置OpenAI API密钥。请在环境变量中设置LLM_API_KEY。"

        prompt = f"""你是一位专业的植物病理学专家。请根据以下信息，提供详细的植物病害治理建议：

植物种类：{plant_class}
病害名称：{disease_name}
患病程度：{disease_degree}
健康状况：{health_status}

请提供：
1. 病害的简要说明
2. 具体的治理措施（包括化学防治、生物防治、农业防治等）
3. 预防措施
4. 注意事项

请用中文回答，内容要专业、实用、易懂。"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一位专业的植物病理学专家，擅长提供植物病害诊断和治理建议。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
            }

            response = requests.post(self.api_base, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"获取治理建议时出错：{str(e)}"


class BaiduWenxinService(LLMService):
    """百度文心一言API服务"""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BAIDU_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("BAIDU_SECRET_KEY", "")
        self.access_token = None
        self._get_access_token()

    def _get_access_token(self):
        """获取百度API访问令牌"""
        if not self.api_key or not self.secret_key:
            return

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }

        try:
            response = requests.post(url, params=params, timeout=10)
            response.raise_for_status()
            self.access_token = response.json().get("access_token")
        except Exception as e:
            print(f"获取百度访问令牌失败：{e}")

    def get_treatment_advice(
        self, plant_class: str, disease_name: str, disease_degree: str, health_status: str
    ) -> str:
        """使用百度文心一言API获取治理建议"""
        if not self.access_token:
            return "错误：未配置百度文心一言API密钥。请设置BAIDU_API_KEY和BAIDU_SECRET_KEY环境变量。"

        prompt = f"""你是一位专业的植物病理学专家。请根据以下信息，提供详细的植物病害治理建议：

植物种类：{plant_class}
病害名称：{disease_name}
患病程度：{disease_degree}
健康状况：{health_status}

请提供：
1. 病害的简要说明
2. 具体的治理措施（包括化学防治、生物防治、农业防治等）
3. 预防措施
4. 注意事项

请用中文回答，内容要专业、实用、易懂。"""

        try:
            url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={self.access_token}"
            headers = {"Content-Type": "application/json"}
            data = {
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("result", "获取建议失败").strip()
        except Exception as e:
            return f"获取治理建议时出错：{str(e)}"


class AlibabaTongyiService(LLMService):
    """阿里通义千问API服务"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.api_base = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def get_treatment_advice(
        self, plant_class: str, disease_name: str, disease_degree: str, health_status: str
    ) -> str:
        """使用阿里通义千问API获取治理建议"""
        if not self.api_key:
            return "错误：未配置阿里通义千问API密钥。请在环境变量中设置DASHSCOPE_API_KEY（推荐）或LLM_API_KEY。"

        prompt = f"""你是一位专业的植物病理学专家。请根据以下信息，提供详细的植物病害治理建议：

植物种类：{plant_class}
病害名称：{disease_name}
患病程度：{disease_degree}
健康状况：{health_status}

请提供：
1. 病害的简要说明
2. 具体的治理措施（包括化学防治、生物防治、农业防治等）
3. 预防措施
4. 注意事项

请用中文回答，内容要专业、实用、易懂。"""

        result = None
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "qwen-turbo",
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            }

            response = requests.post(self.api_base, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # 处理不同的响应格式
            content = None
            if "output" in result and "choices" in result["output"]:
                # 标准格式：result["output"]["choices"][0]["message"]["content"]
                content = result["output"]["choices"][0]["message"]["content"]
            elif "output" in result and "text" in result["output"]:
                # 备选格式：result["output"]["text"]
                content = result["output"]["text"]
            elif "choices" in result:
                # 直接包含choices
                content = result["choices"][0]["message"]["content"]
            elif "text" in result:
                # 直接包含text
                content = result["text"]
            
            if content:
                return content.strip()
            else:
                # 如果都不匹配，返回完整响应用于调试
                return f"API响应格式异常，请检查。响应内容：{str(result)[:500]}"
                
        except KeyError as e:
            # 键错误，返回详细错误信息
            error_msg = f"获取治理建议时出错：API响应格式不符合预期。错误：{str(e)}"
            if result:
                error_msg += f"，响应：{str(result)[:500]}"
            return error_msg
        except Exception as e:
            error_msg = f"获取治理建议时出错：{str(e)}"
            if result:
                error_msg += f"，响应：{str(result)[:500]}"
            return error_msg


def get_llm_service(provider: str = "mock") -> LLMService:
    """
    根据提供商名称获取对应的LLM服务实例
    
    Args:
        provider: 提供商名称，可选值：'openai', 'baidu', 'alibaba', 'mock'
        
    Returns:
        LLMService实例
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIService()
    elif provider == "baidu":
        return BaiduWenxinService()
    elif provider == "alibaba":
        return AlibabaTongyiService()
    elif provider == "mock":
        # 用于测试的模拟服务
        class MockService(LLMService):
            def get_treatment_advice(self, plant_class, disease_name, disease_degree, health_status):
                return f"""针对{plant_class}的{disease_name}（{disease_degree}），建议如下：

1. 病害说明：
   {disease_name}是{plant_class}常见的病害之一，主要影响植物的叶片和生长。

2. 治理措施：
   - 化学防治：使用合适的杀菌剂进行喷洒
   - 生物防治：引入有益微生物或天敌
   - 农业防治：及时清除病叶，改善通风条件

3. 预防措施：
   - 定期检查植物健康状况
   - 保持适宜的湿度和温度
   - 合理施肥，增强植物抗病能力

4. 注意事项：
   - 根据患病程度调整用药浓度
   - 注意用药安全，避免对环境和人体造成危害
   - 如病情严重，建议咨询专业农技人员

（注：这是模拟建议，实际使用时请配置真实的大模型API）"""

        return MockService()
    else:
        raise ValueError(f"不支持的提供商：{provider}")

