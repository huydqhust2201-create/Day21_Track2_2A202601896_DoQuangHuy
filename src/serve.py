from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
import joblib
import os

app = FastAPI()

STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]  # dat trong systemd service
ARTIFACT_CONTAINER = os.environ["ARTIFACT_CONTAINER"]  # dat trong systemd service
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """Tai model.joblib tu Azure Blob Storage ve may khi server khoi dong."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = client.get_blob_client(container=ARTIFACT_CONTAINER, blob=MODEL_KEY)
    with open(MODEL_PATH, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print(f"Da tai model tu {ARTIFACT_CONTAINER}/{MODEL_KEY} ve {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """GitHub Actions goi endpoint nay sau khi trien khai de xac nhan server song."""
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Dau vao:  JSON {"features": [f1, f2, ..., f10]}
    Dau ra:   JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}
    """
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail=f"Can dung 10 dac trung, nhan duoc {len(req.features)}",
        )

    pred = model.predict([req.features])[0]
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    return {"prediction": int(pred), "label": label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
