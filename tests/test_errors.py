from plant_disease.errors import (
    PlantDiseaseError,
    InferenceError,
    LLMServiceError,
    LLMConfigError,
)


def test_hierarchy():
    assert issubclass(InferenceError, PlantDiseaseError)
    assert issubclass(LLMServiceError, PlantDiseaseError)
    assert issubclass(LLMConfigError, LLMServiceError)


def test_can_be_raised_with_message():
    err = LLMConfigError("missing key")
    assert str(err) == "missing key"
