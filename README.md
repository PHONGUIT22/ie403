# IE403 - Multimodal Aspect-Based Sentiment Analysis

Dự án môn học **IE403** (Trường Đại học Công nghệ Thông tin - ĐHQG-HCM / UIT).  
Nghiên cứu và thực nghiệm mô hình phân loại sắc thái tình cảm theo khía cạnh đa phương tiện (**Multimodal Aspect-Based Sentiment Analysis**) kết hợp giữa hình ảnh và văn bản tiếng Việt bằng cách sử dụng kiến trúc **ConvNeXt-Tiny** kết hợp **PhoBERT**.

---

## 📌 Tổng quan kiến trúc & Phương pháp

- **Văn bản (Text)**: Sử dụng mô hình tiền huấn luyện **PhoBERT** (`vinai/phobert-base`) để mã hóa và trích xuất đặc trưng ngữ nghĩa của các đánh giá / bình luận bằng tiếng Việt.
- **Hình ảnh (Vision)**: Sử dụng mạng nơ-ron tích chập **ConvNeXt-Tiny** trích xuất đặc trưng hình ảnh.
- **Phân loại đa nhãn (Multilabel Classification)**: 18 lớp tương ứng với tích phân phối giữa 6 nhóm khía cạnh (Aspects) và 3 sắc thái cảm xúc (Sentiments):
  - *Aspects*: `Facilities`, `Food`, `Location`, `Public_area`, `Room`, `Service`.
  - *Sentiments*: `Negative`, `Neutral`, `Positive`.

---

## 📁 Cấu trúc thư mục

```text
IE403/
├── DoAn/                               # Báo cáo đồ án, kịch bản thuyết trình môn học
├── seminar/                            # Slide và bài báo seminar
├── paper_final/                        # Bản thảo bài báo khoa học theo template IEEE
├── train.json                          # Tập dữ liệu huấn luyện (annotations)
├── dev.json                            # Tập dữ liệu kiểm định (annotations)
├── test.json                           # Tập dữ liệu kiểm thử (annotations)
├── roi_data.csv                        # Dữ liệu trích xuất đặc trưng vùng
├── all_image_label.xlsx                # Bảng thống kê toàn bộ nhãn ảnh
├── preprocessing.py                    # Script tiền xử lý dữ liệu và tạo DataLoader
├── train_and_eval.py                   # Script huấn luyện và đánh giá mô hình
├── test.py                             # Script đánh giá mô hình trên tập kiểm thử
├── plot_f1_category.py                 # Script trực quan hóa F1-Score theo nhóm chủ đề
├── plot_f1_sentiment.py                # Script trực quan hóa F1-Score theo nhóm cảm xúc
├── ConvNeXt_Tiny.ipynb                 # Notebook thực nghiệm mô hình ConvNeXt-Tiny
├── ConvNeXt_Tiny_phoBERT.ipynb         # Notebook thực nghiệm mô hình đa phương thức ConvNeXt + PhoBERT
├── aspect_sentiment_train.ipynb        # Notebook chi tiết quá trình huấn luyện Aspect#Sentiment
├── requirements.txt                    # Danh sách thư viện cần thiết
└── README.md
```

---

## 💾 Dữ liệu & Trọng số mô hình (Dataset & Pretrained Weights)

Do giới hạn dung lượng lưu trữ của GitHub (tối đa 100MB/file), tập ảnh gốc và file trọng số huấn luyện được lưu trữ ngoài:

1. **Tập dữ liệu hình ảnh (`image/`)**:
   - Gồm ~18.310 ảnh tương ứng với các mẫu dữ liệu.
   - 🔗 **Link tải tập ảnh (Google Drive)**: *[Điền link Google Drive của bạn vào đây]*
   - *Sau khi tải về, giải nén thư mục và đặt tên là `image/` ngay tại thư mục gốc của dự án.*

2. **File trọng số mô hình (`multimodal_model.pth`)**:
   - Dung lượng: ~654 MB.
   - 🔗 **Link tải model weight (Google Drive)**: *[Điền link Google Drive của bạn vào đây]*
   - *Tải về và đặt file `multimodal_model.pth` ngay tại thư mục gốc để script kiểm thử `test.py` có thể nạp mô hình.*

---

## 🚀 Cài đặt và Hướng dẫn sử dụng

### 1. Cài đặt môi trường

Khuyến nghị sử dụng Python 3.10+ và tạo môi trường ảo:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Tiền xử lý dữ liệu

Chạy script tiền xử lý dữ liệu:
```bash
python preprocessing.py
```

### 3. Huấn luyện mô hình

Chạy quá trình huấn luyện kết hợp ConvNeXt-Tiny và PhoBERT:
```bash
python train_and_eval.py
```

### 4. Đánh giá trên tập kiểm thử

Chạy kiểm thử với file trọng số đã lưu:
```bash
python test.py
```

### 5. Vẽ biểu đồ trực quan hóa kết quả

```bash
python plot_f1_category.py
python plot_f1_sentiment.py
```
