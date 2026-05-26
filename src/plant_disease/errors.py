"""Custom exception hierarchy for plant_disease."""


class PlantDiseaseError(Exception):
    """Base exception for all plant_disease errors."""


class InferenceError(PlantDiseaseError):
    """Raised when inference fails (weights load, forward pass, image decode)."""


class LLMServiceError(PlantDiseaseError):
    """Raised when an LLM provider call fails (network, timeout, HTTP error)."""


class LLMConfigError(LLMServiceError):
    """Raised when an LLM provider is missing required configuration."""
