"""Deterministic mock provider for tests and missing-key fallback."""

from plant_disease.llm.base import LLMService


class MockProvider(LLMService):
    def get_treatment_advice(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> str:
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
   - 注意用药安全
   - 如病情严重，建议咨询专业农技人员

（注：这是模拟建议，实际使用时请配置真实的大模型 API）"""
