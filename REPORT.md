# Báo cáo — Vòng đời mô hình ML trong sản xuất

**Repo:** https://github.com/huydqhust2201-create/Day21_Track2_2A202601896_DoQuangHuy

**Ảnh chụp màn hình bằng chứng:** xem thư mục [`screenshots/`](screenshots/)
1. [`01-mlflow-experiments.png`](screenshots/01-mlflow-experiments.png) — MLflow UI, 7 run với f1_score/accuracy khác nhau
2. [`02-github-actions-4-jobs-green.png`](screenshots/02-github-actions-4-jobs-green.png) — 4 job Actions đều xanh
3. [`03-endpoint-curl-results.png`](screenshots/03-endpoint-curl-results.png) — curl `/healthz` và `/score` trên EC2
4. [`04-azure-storage-container.png`](screenshots/04-azure-storage-container.png) — container `labstore` với `dvc/` và `artifacts/`
5. [`05-quality-gate-blocked.png`](screenshots/05-quality-gate-blocked.png) — Quality Gate chặn thật, Release bị skip

## 1. Bộ siêu tham số đã chọn và lý do

Ba thí nghiệm chạy trên `train_batch1.csv` (22361 mẫu), đánh giá trên `holdout.csv` (500 mẫu), ghi lại trong MLflow:

| n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|
| 100 | 0.1 | 3 | 0.7032 | 0.8700 |
| 50 | 0.05 | 2 | **0.6337** (< 0.65) | 0.8520 |
| **200** | **0.1** | **5** | **0.7574** | 0.8860 |

Chọn bộ **n_estimators=200, learning_rate=0.1, max_depth=5** vì cho F1 cao nhất và vượt ngưỡng 0.65.

## 2. Vì sao ngưỡng chất lượng đặt trên F1, không phải accuracy

Dữ liệu Adult mất cân bằng lớp: chỉ **24.8%** mẫu thuộc lớp thu nhập cao. Một mô hình luôn đoán "thu nhập thấp" đạt accuracy 0.752 nhưng F1 = 0 — không bắt được trường hợp nào.

Số liệu thật từ 3 thí nghiệm minh chứng điều này: **accuracy chỉ dao động 0.852–0.886 (chênh 0.034)** trong khi **F1 dao động 0.634–0.757 (chênh 0.124)** — biên độ gấp **~3.6 lần**. Accuracy gần như "vô cảm" trước chất lượng thật của mô hình vì lớp đa số (75.2%) luôn kéo điểm số lên cao bất kể mô hình có học được lớp thiểu số hay không; F1 của lớp dương phản ánh đúng khả năng mô hình phát hiện đúng nhóm cần quan tâm (thu nhập cao). Nếu đặt ngưỡng trên accuracy, bộ tham số yếu nhất (accuracy 0.852) vẫn dễ dàng "qua cổng" dù gần như vô dụng với lớp thiểu số.

## 3. So sánh F1 giữa 22361 mẫu và 44722 mẫu

| Lượt chạy | Số mẫu train | f1_score | accuracy |
|---|---|---|---|
| Lần đầu | 22361 | 0.7574 | 0.8860 |
| Sau khi bổ sung `train_batch2` | 44722 | **0.7619** | 0.8900 |

F1 chỉ tăng nhẹ **+0.0045**. `train_batch2.csv` là nửa còn lại của cùng bộ Adult Census Income, được chia ngẫu nhiên từ cùng một nguồn nên có cùng phân phối với `train_batch1.csv`. Với siêu tham số cố định, mô hình GradientBoosting đã học gần hết những gì có thể học được từ 22361 mẫu đầu; gấp đôi dữ liệu cùng phân phối không mang lại nhiều thông tin mới nên chỉ số gần như đi ngang. Điều thực sự được kiểm chứng ở bước này không phải là chỉ số tăng vọt, mà là **quy trình chạy đúng đầu-cuối**: một commit dữ liệu duy nhất (`data: bo sung 22361 mau du lieu moi`) tự động kích hoạt lại pipeline, tên lần chạy trên tab Actions trùng khớp với commit message, chứng minh huấn luyện liên tục hoạt động mà không cần thao tác tay.

## 4. Khó khăn gặp phải và cách giải quyết

- **Subscription Azure của trường (tenant vinuni.edu.vn) chặn tạo VM ở mọi vùng và mọi kích thước** (lỗi `SkuNotAvailable: NotAvailableForSubscription`), dù quota vCPU vẫn còn 10 — xác nhận qua cả Azure CLI lẫn Azure Portal ở 4 vùng khác nhau (southeastasia, eastus, japaneast, koreacentral). Đây là giới hạn chính sách của subscription giáo dục, không phải lỗi cấu hình.
- **Tài khoản Azure free trial cá nhân bị từ chối** ("You're not eligible for an Azure free account"). Chuyển sang đăng ký Pay-As-You-Go — nhưng subscription mới lại gặp tiếp: quota Compute = 0, resource provider `Microsoft.Compute`/`Microsoft.Quota` chưa đăng ký, và yêu cầu tăng quota tự phục vụ (cả qua Portal lẫn CLI) đều trả về lỗi `ContactSupport` chung chung — vượt quá khả năng tự xử lý vì gói support là Free.
- **Đào sâu Azure Portal phát hiện nguyên nhân gốc thật sự**: family `Standard BS` (chứa B1s/B2s) đã bị Azure liệt vào **legacy, khoá tăng quota vĩnh viễn** cho subscription mới ("Quota increases are disabled for this legacy SKU"). Đây là lý do mọi vùng đều thất bại giống hệt nhau suốt từ đầu — không phải lỗi cấu hình hay giới hạn tài khoản, mà do chọn nhầm một dòng SKU đã bị khai tử. Thử với family thay thế **Bsv2** (`Standard_B2s_v2`) thì lỗi đổi từ `SkuNotAvailable` (không sửa được) sang `QuotaExceeded` (loại tự phục vụ được) — nhưng yêu cầu tăng quota cho Bsv2 vẫn tiếp tục `Failed` trong giới hạn thời gian của bài lab.
- **Quyết định chuyển hẳn sang AWS EC2** để không phụ thuộc vào việc Azure duyệt quota. Gặp thêm một trở ngại tương tự: thẻ Visa mặc định không xác minh được thanh toán AWS — thêm thẻ Mastercard thứ hai thì qua. Sau đó tạo IAM user với quyền `AmazonEC2FullAccess`, cấu hình `aws configure`, và triển khai thành công EC2 `t3.micro` (Ubuntu 22.04) tại `ap-southeast-1`, tái sử dụng đúng cặp SSH key (`income_deploy`) đã tạo từ đầu cho Azure nên không cần đổi secret `SERVER_SSH_KEY`. Dữ liệu và model vẫn giữ nguyên trên Azure Blob Storage — EC2 chỉ cần `AZURE_STORAGE_CONNECTION_STRING` để tải model, không cần di chuyển gì giữa hai cloud.

## Bằng chứng đã có

- MLflow UI: 3 lần chạy với siêu tham số và chỉ số khác nhau.
- Tab Actions: **cả 4 job xanh** (Unit Test → Train → Quality Gate → Release) ở run [`32493410617`](https://github.com/huydqhust2201-create/Day21_Track2_2A202601896_DoQuangHuy/actions/runs/32493410617), cùng nhiều lần chạy trước đó cho 3 job đầu (lần đầu tiên, lần kích hoạt bởi commit dữ liệu `b3011d4`, và lần khôi phục tham số tốt `4ab4b63`).
- **Quality gate thực sự chặn triển khai**: run [`32484353299`](https://github.com/huydqhust2201-create/Day21_Track2_2A202601896_DoQuangHuy/actions/runs/32484353299) dùng tham số cố tình yếu (n_estimators=10, learning_rate=0.01, max_depth=1) cho **f1_score=0.0000, accuracy=0.7420** — đúng kịch bản "mô hình đoán bừa" ở mục 4.2: accuracy trông cao nhưng F1 bằng 0. Job Quality Gate FAIL, job Release **bị skip hoàn toàn**, không có thao tác tay nào can thiệp.
- Cloud Storage: dữ liệu dưới prefix `dvc/`, model tại `artifacts/current/model.joblib` trong container `labstore` (Azure Blob Storage).
- **Endpoint đang sống trên EC2** (`http://18.138.232.210:8080`):
  - `curl /healthz` → `{"status":"ok"}`
  - `curl -X POST /score -d '{"features":[60,2,5,2,4,0,1,0,0,45]}'` → `{"prediction":0,"label":"thu_nhap_thap"}`
  - `curl -X POST /score -d '{"features":[28,2,14,2,11,0,1,0,0,45]}'` → `{"prediction":1,"label":"thu_nhap_cao"}`
