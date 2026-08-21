# Báo cáo — Vòng đời mô hình ML trong sản xuất

**Repo:** https://github.com/huydqhust2201-create/Day21_Track2_2A202601896_DoQuangHuy

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

- **Subscription Azure của trường (tenant vinuni.edu.vn) chặn tạo VM ở mọi vùng và mọi kích thước** (lỗi `SkuNotAvailable: NotAvailableForSubscription`), dù quota vCPU vẫn còn 10 — xác nhận qua cả Azure CLI lẫn Azure Portal ở 4 vùng khác nhau (southeastasia, eastus, japaneast, koreacentral). Đây là giới hạn chính sách của subscription giáo dục, không phải lỗi cấu hình. Cách giải quyết: chuyển sang tài khoản Azure Pay-As-You-Go cá nhân.
- **Tài khoản Azure free trial cá nhân bị từ chối** ("You're not eligible for an Azure free account") — có thể do số điện thoại/thiết bị đã được Microsoft ghi nhận từng dùng free trial. Chuyển sang đăng ký Pay-As-You-Go (có tính phí thật nhưng rất nhỏ cho vài giờ chạy VM B1s).
- **Subscription Pay-As-You-Go mới có quota Compute = 0 và resource provider `Microsoft.Compute` chưa đăng ký** — hành vi bình thường với subscription vừa tạo, cần thời gian để Azure xác minh (có thể vài giờ). Đã gửi yêu cầu đăng ký provider và xin tăng quota; ba job đầu của pipeline (Unit Test, Train, Quality Gate) đã được xác minh chạy xanh nhiều lần trong lúc chờ, job Release sẽ hoàn tất ngay khi VM tạo được.

## Bằng chứng đã có

- MLflow UI: 3 lần chạy với siêu tham số và chỉ số khác nhau.
- Tab Actions: Unit Test → Train → Quality Gate xanh ở nhiều lần chạy (lần đầu, lần kích hoạt bởi commit dữ liệu `b3011d4`, và lần khôi phục tham số tốt `4ab4b63`).
- **Quality gate thực sự chặn triển khai**: run [`32484353299`](https://github.com/huydqhust2201-create/Day21_Track2_2A202601896_DoQuangHuy/actions/runs/32484353299) dùng tham số cố tình yếu (n_estimators=10, learning_rate=0.01, max_depth=1) cho **f1_score=0.0000, accuracy=0.7420** — đúng kịch bản "mô hình đoán bừa" ở mục 4.2: accuracy trông cao nhưng F1 bằng 0. Job Quality Gate FAIL, job Release **bị skip hoàn toàn** (không chạy), không có thao tác tay nào can thiệp.
- Cloud Storage: dữ liệu dưới prefix `dvc/`, model tại `artifacts/current/model.joblib` trong container `labstore`.
- Job Release: đang chờ VM (đã xác định nguyên nhân là giới hạn xác minh tài khoản mới của Azure, đang chờ xử lý).
