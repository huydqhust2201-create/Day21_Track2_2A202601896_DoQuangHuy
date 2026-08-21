# mlops-lab

Vòng đời mô hình ML trong sản xuất: thí nghiệm cục bộ (MLflow) → phiên bản hoá dữ liệu (DVC)
→ CI/CD (GitHub Actions) → triển khai (VM + systemd) → huấn luyện liên tục.

## Đã hoàn thành (chạy được ngay trên máy)

- `prepare_data.py` — tải và xử lý bộ Adult/Census Income, sinh `data/train_batch1.csv`
  (22361 mẫu), `data/holdout.csv` (500 mẫu), `data/train_batch2.csv` (22361 mẫu).
- `append_batch.py` — mô phỏng bổ sung dữ liệu mới (nối `train_batch2` vào `train_batch1`).
- `src/train.py` — huấn luyện `GradientBoostingClassifier`, ghi `f1_score`/`accuracy` vào
  MLflow, xuất `outputs/report.json` và `models/model.joblib`.
- `src/serve.py` — API FastAPI (`GET /healthz`, `POST /score`), tải `model.joblib` từ Cloud
  Storage lúc khởi động.
- `tests/test_train.py` — 3 unit test chạy trên dữ liệu ảo, không cần cloud.
- `.github/workflows/cicd.yml` — 4 job: Unit Test → Train → Quality Gate → Release.
- `params.yaml`, `requirements.txt`, `.gitignore`.

## Chạy cục bộ

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

python prepare_data.py

export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts

python src/train.py
pytest tests/ -v

mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Chạy ít nhất 3 lần `python src/train.py` với `params.yaml` khác nhau (xem gợi ý bảng
siêu tham số trong đề bài), rồi so sánh trong MLflow UI. Đặt bộ tham số tốt nhất
(f1_score >= 0.65) vào `params.yaml` trước khi sang phần cloud.

## Những gì còn lại — cần tài khoản cloud/GitHub thật của bạn

Tôi không thể tự tạo bucket, VM, hay GitHub secrets thay bạn (cần thông tin xác thực
thật và có thể phát sinh chi phí). Dưới đây là lệnh đầy đủ, đúng thứ tự, bạn copy-paste
vào terminal đã đăng nhập `gcloud` (mặc định lấy GCP làm ví dụ — xem bảng ánh xạ
AWS/Azure trong đề bài nếu dùng provider khác).

### 1. Tạo repo GitHub và đẩy code hiện tại

```bash
git init
git add .
git commit -m "feat: initial mlops-lab scaffold"
git branch -M main
git remote add origin <URL_REPO_CUA_BAN>
git push -u origin main
```

### 2. Bucket + service account + DVC

```bash
export PROJECT=<YOUR_PROJECT>
export BUCKET=<BUCKET_NAME>

gsutil mb -p $PROJECT -l us-central1 gs://$BUCKET
gcloud services enable storage.googleapis.com --project $PROJECT

gcloud iam service-accounts create income-lab-sa \
  --display-name "Income Lab SA" --project $PROJECT

gsutil iam ch \
  serviceAccount:income-lab-sa@$PROJECT.iam.gserviceaccount.com:roles/storage.objectAdmin \
  gs://$BUCKET

gcloud iam service-accounts keys create sa-key.json \
  --iam-account income-lab-sa@$PROJECT.iam.gserviceaccount.com

pip install "dvc[gs]"
dvc init
dvc remote add -d labstore gs://$BUCKET/dvc
dvc remote modify labstore credentialpath sa-key.json

dvc add data/train_batch1.csv
dvc add data/holdout.csv
dvc add data/train_batch2.csv

git add data/train_batch1.csv.dvc data/holdout.csv.dvc data/train_batch2.csv.dvc \
        .gitignore .dvc/config
git commit -m "feat: track datasets with DVC"

dvc push
```

Kiểm chứng: Cloud Storage Console phải thấy dữ liệu dưới prefix `dvc/`.

### 3. Tạo VM và cài server suy luận

```bash
gcloud compute instances create income-api \
  --zone=us-central1-a --machine-type=e2-small \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --tags=income-api --project $PROJECT

gcloud compute firewall-rules create allow-income-api \
  --allow=tcp:8080 --target-tags=income-api --project $PROJECT

gcloud compute instances describe income-api --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

SSH vào VM và cài gói:

```bash
gcloud compute ssh income-api --zone=us-central1-a
# Bên trong VM:
sudo apt update && sudo apt install -y python3-pip
pip3 install fastapi uvicorn scikit-learn joblib google-cloud-storage
mkdir -p ~/models ~/src
exit
```

Copy khoá và code lên VM:

```bash
gcloud compute scp sa-key.json income-api:~/sa-key.json --zone=us-central1-a
gcloud compute scp src/serve.py income-api:~/src/serve.py --zone=us-central1-a
```

Tạo systemd service (SSH vào VM, thay `<YOUR_BUCKET_NAME>` và `USER`):

```bash
gcloud compute ssh income-api --zone=us-central1-a
```

```bash
sudo tee /etc/systemd/system/income-api.service > /dev/null <<'EOF'
[Unit]
Description=Income Model Inference Server
After=network.target

[Service]
User=USER
WorkingDirectory=/home/USER
Environment="ARTIFACT_BUCKET=<YOUR_BUCKET_NAME>"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/USER/sa-key.json"
ExecStart=/usr/bin/python3 /home/USER/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable income-api
```

Chưa `systemctl start` lúc này — mô hình chưa tồn tại trên cloud storage cho tới khi
pipeline chạy lần đầu (bước 5).

### 4. SSH key + GitHub Secrets

```bash
ssh-keygen -t ed25519 -f ~/.ssh/income_deploy -N "" -C "github-actions-deploy"

gcloud compute ssh income-api --zone=us-central1-a \
  --command "echo '$(cat ~/.ssh/income_deploy.pub)' >> ~/.ssh/authorized_keys"
```

Trong repo GitHub: **Settings → Secrets and variables → Actions → New repository secret**,
thêm đúng 5 secret:

| Tên secret | Giá trị |
|---|---|
| `STORAGE_CREDENTIALS` | Toàn bộ nội dung `sa-key.json` |
| `ARTIFACT_BUCKET` | Tên bucket |
| `SERVER_HOST` | IP công khai của VM |
| `SERVER_USER` | `echo $USER` trong session SSH |
| `SERVER_SSH_KEY` | Toàn bộ nội dung `~/.ssh/income_deploy` |

### 5. Chạy pipeline lần đầu

```bash
git add .
git commit -m "feat: add CI/CD pipeline, tests, and serving API"
git push origin main
```

Theo dõi tab **Actions**. Sau khi cả 4 job xanh:

```bash
gcloud compute ssh income-api --zone=us-central1-a \
  --command "sudo systemctl start income-api"

VM_IP=<YOUR_VM_IP>
curl http://$VM_IP:8080/healthz
curl -X POST http://$VM_IP:8080/score \
  -H "Content-Type: application/json" \
  -d '{"features": [60, 2, 5, 2, 4, 0, 1, 0, 0, 45]}'
```

### 6. Huấn luyện liên tục với dữ liệu mới

```bash
python append_batch.py

dvc add data/train_batch1.csv
git add data/train_batch1.csv.dvc
git commit -m "data: bo sung du lieu moi (train_batch2)"

dvc push          # LUÔN đứng trước git push
git push origin main
```

### 7. Kiểm chứng quality gate thật sự chặn

Tạm sửa `params.yaml` về bộ tham số yếu (`n_estimators: 50, learning_rate: 0.05,
max_depth: 2`), push, xác nhận job **Quality Gate** đỏ và **Release** không chạy. Sau đó
trả lại tham số tốt và push lại.
