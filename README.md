# 📊 Bài Tập Lớn Số 2 — Môn Python 🐍

**Tên đề tài:** Nhận dạng ảnh CIFAR-10 bằng Mạng Nơ-ron Tích Chập (CNN 3 lớp)

## 📑 Mục lục

📁 Report/  
📄 Báo-cáo.pdf # Báo cáo tổng hợp kết quả bài tập lớn

📁 SourceCode/  
📁 CNN_Res/ # Kết quả huấn luyện: Loss Curve, Accuracy Curve, Confusion Matrix  
📁 CNN_Code/ # Code

📄 README.md # File mô tả dự án

## 📝 MÔ TẢ DỰ ÁN

Dự án này tập trung vào **xây dựng và huấn luyện một mạng CNN gồm 3 lớp convolution** để phân loại ảnh trên tập dữ liệu **CIFAR-10** (60.000 ảnh màu 32×32, thuộc 10 lớp).  
Trong quá trình thực hiện, mô hình sẽ được cải tiến với các kỹ thuật như **Batch Normalization**, **Dropout**, **Data Augmentation** và **Learning-rate Scheduler** nhằm nâng cao độ chính xác và giảm hiện tượng overfitting.

---

## 📌 CÁC BƯỚC THỰC HIỆN

- **Bước 1:** Định nghĩa kiến trúc CNN  
  - Xây dựng một mạng CNN gồm 3 khối convolution:  
    > - Mỗi khối: Conv2d → BatchNorm2d → ReLU → MaxPool2d  
    > - Số kênh tăng dần qua các lớp: 3 → 64 → 128 → 256  
    > - Thêm lớp Dropout (p=0.5) trước fully connected để giảm overfitting  
  - Mã lệnh định nghĩa model và giải thích từng thành phần.

- **Bước 2:** Chuẩn bị dữ liệu với Data Augmentation  
  - Sử dụng **RandomCrop(padding=4)** và **RandomHorizontalFlip** trên tập train để tăng tính đa dạng.  
  - Chỉ áp dụng **ToTensor()** và **Normalize** (mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]) cho train/validation/test.

- **Bước 3:** Huấn luyện và Validate mô hình  
  - Khởi tạo model lên device (GPU nếu có).  
  - Sử dụng **CrossEntropyLoss** làm hàm mất mát, **Adam** làm optimizer (learning rate = 1e-3).  
  - Áp dụng **StepLR** (gamma = 0.5, step_size = 10) để giảm learning rate mỗi 10 epoch.  
  - Huấn luyện trong **25 epoch**:  
    > 1. Chuyển `model.train()`, lặp qua từng batch, tính toán loss, backward và optimizer.step().  
    > 2. Chuyển `model.eval()` trên tập validation, không tính gradient, tính `val_loss` và `val_acc`.  
    > 3. Lưu lịch sử `train_loss`, `val_loss`, `train_acc`, `val_acc` vào cấu trúc `history`.

- **Bước 4:** Đánh giá trên tập Test và trực quan hóa  
  - Sau khi huấn luyện xong, chuyển `model.eval()` và chạy dự đoán trên toàn bộ test set.  
  - Tính **Test Accuracy**.  
  - Vẽ **Loss Curve** (Train vs. Validation) và **Accuracy Curve** (Train vs. Validation) bằng matplotlib.  
  - Sử dụng **confusion_matrix** từ scikit-learn để tạo ma trận nhầm lẫn, sau đó hiển thị bằng **ConfusionMatrixDisplay**.  
  - Lưu ba hình ảnh:  
    > - `Loss Curve.png`  
    > - `Accuracy Curve.png`  
    > - `Confusion Matrix.png`  

---

## 📂 Cấu trúc thư mục chi tiết

1. **`Report/`**  
   - Bao gồm file PDF “Báo-cáo.pdf” là báo cáo tổng hợp toàn bộ quá trình triển khai, giải thích chi tiết từng bước, kết quả thực thi và nhận xét, đề xuất cải tiến.

2. **`SourceCode/CNN-code.py`**  
   - File Python chính, chứa tất cả các bước từ định nghĩa kiến trúc, chuẩn bị dữ liệu, huấn luyện, validate, đến đánh giá và lưu hình ảnh kết quả.

3. **`SourceCode/CNN_Res/`**  
   - **`Loss Curve.png`**: Đồ thị thể hiện giá trị loss của train và validation qua 25 epoch.  
   - **`Accuracy Curve.png`**: Đồ thị thể hiện giá trị accuracy của train và validation qua 25 epoch.  
   - **`Confusion Matrix.png`**: Ma trận nhầm lẫn trên tập test, trực quan mức độ đúng/sai giữa 10 lớp CIFAR-10.

4. **`SourceCode/README.md`**  
   - File này (README.md) mô tả chi tiết cấu trúc thư mục và hướng dẫn chạy project.

---

