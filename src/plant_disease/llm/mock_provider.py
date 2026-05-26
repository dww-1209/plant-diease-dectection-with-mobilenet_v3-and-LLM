"""离线测试 / 无 key 时的兜底 provider。

不发任何 HTTP 请求，按段 yield 一段固定模板文本。两个用途：
1. 跑测试时不依赖外部 API 与网络
2. 用户没填任何 key 时，前端依然能看到一个合理的"建议"长什么样
"""

from __future__ import annotations

from collections.abc import Iterator

from plant_disease.llm.base import LLMService


class MockProvider(LLMService):
    """不调真实 API，按段产出模板文本。"""

    def stream_treatment_advice(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> Iterator[str]:
        sections = [
            f"针对{plant_class}的{disease_name}（{disease_degree}），建议如下：\n\n",
            "1. 病害说明：\n"
            f"   {disease_name}是{plant_class}常见的病害之一，主要影响植物的叶片和生长。\n\n",
            "2. 治理措施：\n"
            "   - 化学防治：使用合适的杀菌剂进行喷洒\n"
            "   - 生物防治：引入有益微生物或天敌\n"
            "   - 农业防治：及时清除病叶，改善通风条件\n\n",
            "3. 预防措施：\n"
            "   - 定期检查植物健康状况\n"
            "   - 保持适宜的湿度和温度\n"
            "   - 合理施肥，增强植物抗病能力\n\n",
            "4. 注意事项：\n"
            "   - 根据患病程度调整用药浓度\n"
            "   - 注意用药安全\n"
            "   - 如病情严重，建议咨询专业农技人员\n\n",
            "（注：这是模拟建议，实际使用时请配置真实的大模型 API）",
        ]
        yield from sections
