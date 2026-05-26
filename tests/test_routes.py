from io import BytesIO


def test_get_treatment_advice_with_mock(client):
    resp = client.post(
        "/get_treatment_advice",
        json={
            "plant_class": "番茄",
            "disease_name": "早疫病",
            "disease_degree": "一般",
            "health_status": "患病",
            "provider": "mock",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "番茄" in data["advice"]


def test_get_treatment_advice_missing_fields(client):
    resp = client.post("/get_treatment_advice", json={"plant_class": "", "disease_name": ""})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_predict_no_file(client):
    resp = client.post("/predict")
    assert resp.status_code in (400, 503)


def test_predict_returns_503_when_init_failed(client):
    # 由于权重不存在，create_app 应把 InferenceModel 初始化失败保存下来；
    # /predict 在没有 image 字段时也走 400，但模型初始化失败的情况下应优先返回 503。
    resp = client.post(
        "/predict",
        data={"image": (BytesIO(b"x"), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 503
    assert "未初始化" in resp.get_json()["message"]


def test_get_treatment_advice_invalid_provider(client):
    resp = client.post(
        "/get_treatment_advice",
        json={
            "plant_class": "a", "disease_name": "b",
            "disease_degree": "c", "health_status": "d",
            "provider": "no-such",
        },
    )
    assert resp.status_code == 400
