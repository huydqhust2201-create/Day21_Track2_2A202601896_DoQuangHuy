from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage  # thay bang SDK cua provider da chon (boto3 / azure-storage-blob)
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]  # duoc dat trong systemd service
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """Tai model.joblib tu cloud storage ve may khi server khoi dong."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(ARTIFACT_BUCKET)
    blob = bucket.blob(MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print(f"Da tai model tu gs://{ARTIFACT_BUCKET}/{MODEL_KEY} ve {MODEL_PATH}")


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
