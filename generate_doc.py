
import docx
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

# Tạo tài liệu Word
document = Document()

# Thêm tiêu đề chính
title = document.add_heading('Phương pháp', level=1)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Thêm đoạn giới thiệu
intro = document.add_paragraph(
    "Để giải quyết bài toán phân loại đa nhãn trên tập dữ liệu hình ảnh, chúng tôi sử dụng mô hình "
    "ConvNeXt-Tiny với trọng số tiền huấn luyện (pre-trained weights) và tinh chỉnh (fine-tuned) trên "
    "tập dữ liệu tùy chỉnh. Quy trình triển khai bao gồm các bước: tiền xử lý dữ liệu, khởi tạo mô hình, "
    "huấn luyện, và đánh giá. Dưới đây là mô tả chi tiết từng bước, dựa trên mã nguồn được cung cấp."
)

# Thêm tiêu đề cấp 2: Tiền xử lý dữ liệu
document.add_heading('Tiền xử lý dữ liệu', level=2)
document.add_paragraph(
    "Tập dữ liệu được chia thành ba tập con: tập huấn luyện (5.428 mẫu), tập phát triển (1.789 mẫu), và "
    "tập kiểm tra (1.841 mẫu). Mỗi mẫu gồm một hình ảnh và các nhãn tương ứng thuộc sáu lớp: Location, "
    "Food, Room, Facilities, Service, và Public_area. Các nhãn được biểu diễn dưới dạng nhị phân (0 hoặc 1) "
    "cho bài toán phân loại đa nhãn. Phân bố mẫu theo nhãn trong tập huấn luyện bao gồm: Location (0 mẫu), "
    "Food (321 mẫu), Room (2.357 mẫu), Facilities (612 mẫu), Service (57 mẫu), và Public_area (3.096 mẫu)."
)
document.add_paragraph(
    "Dữ liệu được tải bằng các đối tượng Dataset tùy chỉnh (train_dataset, dev_dataset, test_dataset) từ "
    "file preprocessing.py. Các DataLoader được tạo với kích thước batch là 32, trong đó tập huấn luyện "
    "được xáo trộn (shuffle=True) để tăng tính ngẫu nhiên, trong khi tập phát triển và kiểm tra giữ nguyên thứ tự."
)

# Thêm tiêu đề cấp 2: Khởi tạo mô hình
document.add_heading('Khởi tạo mô hình', level=2)
document.add_paragraph(
    "Chúng tôi sử dụng kiến trúc ConvNeXt-Tiny [1], một mô hình mạng nơ-ron tích chập (CNN) hiệu quả, với "
    "trọng số tiền huấn luyện từ ConvNeXt_Tiny_Weights.DEFAULT. Lớp phân loại cuối cùng của mô hình được thay "
    "thế bằng một lớp tuyến tính (nn.Linear) với số đầu ra là 6, tương ứng với số lớp trong bài toán. Quy trình "
    "khởi tạo bao gồm:"
)
document.add_paragraph(
    "• Tải mô hình ConvNeXt-Tiny với trọng số tiền huấn luyện.\n"
    "• Thay đổi lớp model.classifier[2] để có 6 đầu ra, phù hợp với bài toán phân loại đa nhãn.",
    style='List Bullet'
)
document.add_paragraph(
    "Mô hình được chuyển sang thiết bị tính toán (GPU nếu khả dụng, ngược lại là CPU) để tối ưu hóa hiệu suất."
)

# Thêm tiêu đề cấp 2: Hàm mất mát và tối ưu hóa
document.add_heading('Hàm mất mát và tối ưu hóa', level=2)
document.add_paragraph(
    "Để xử lý bài toán phân loại đa nhãn, chúng tôi sử dụng hàm mất mát BCEWithLogitsLoss với trọng số "
    "pos_weight để khắc phục vấn đề mất cân bằng lớp. Trọng số pos_weight được tính dựa trên tỷ lệ giữa số "
    "mẫu âm tính và dương tính cho mỗi lớp trong tập huấn luyện, theo công thức:"
)
document.add_paragraph(
    "pos_weight = (số mẫu âm tính) / (số mẫu dương tính + ε),",
    style='Body Text'
)
document.add_paragraph(
    "trong đó ε = 10⁻⁵ để tránh chia cho 0. Các giá trị pos_weight được tính toán bằng cách duyệt qua "
    "train_loader và tổng hợp số mẫu dương tính và âm tính cho từng lớp. Hàm mất mát và thuật toán tối ưu "
    "hóa Adam (với tốc độ học 0.0001) được cấu hình để huấn luyện mô hình."
)

# Thêm tiêu đề cấp 2: Huấn luyện mô hình
document.add_heading('Huấn luyện mô hình', level=2)
document.add_paragraph(
    "Mô hình được huấn luyện trong 5 epoch với các bước sau:"
)
document.add_paragraph(
    "1. Đặt mô hình ở chế độ huấn luyện (model.train()).\n"
    "2. Với mỗi batch trong train_loader:\n"
    "   • Chuyển hình ảnh và nhãn sang thiết bị tính toán (GPU/CPU).\n"
    "   • Thực hiện lan truyền xuôi (forward pass) để tính toán đầu ra (logits).\n"
    "   • Tính toán hàm mất mát bằng BCEWithLogitsLoss với pos_weight.\n"
    "   • Thực hiện lan truyền ngược (backward pass) để tính gradient.\n"
    "   • Cập nhật trọng số mô hình bằng thuật toán Adam.\n"
    "3. Theo dõi tổng giá trị mất mát sau mỗi epoch để đánh giá tiến trình huấn luyện.",
    style='List Number'
)
document.add_paragraph(
    "Thanh tiến trình (tqdm) được sử dụng để hiển thị trạng thái huấn luyện theo thời gian thực."
)

# Thêm tiêu đề cấp 2: Đánh giá mô hình
document.add_heading('Đánh giá mô hình', level=2)
document.add_paragraph(
    "Mô hình được đánh giá trên tập phát triển và tập kiểm tra bằng hai phương pháp:"
)
document.add_paragraph(
    "• Độ chính xác tổng thể (micro accuracy): Được tính bằng cách so sánh dự đoán nhị phân (sử dụng ngưỡng "
    "sigmoid 0.5) với nhãn thực tế trên toàn bộ tập dữ liệu. Công thức là:\n"
    "  Micro Accuracy = (Tổng số dự đoán đúng) / (Tổng số nhãn).\n"
    "• Báo cáo phân loại chi tiết: Sử dụng classification_report từ scikit-learn để tính toán các chỉ số "
    "precision, recall, và F1-score cho từng lớp, cùng với các chỉ số tổng hợp (micro, macro, weighted, và "
    "samples average). Ma trận nhầm lẫn (multilabel_confusion_matrix) được tạo cho từng lớp để phân tích "
    "hiệu suất chi tiết.",
    style='List Bullet'
)
document.add_paragraph(
    "Trên tập phát triển, các dự đoán và nhãn thực tế được thu thập bằng cách duyệt qua dev_loader, áp dụng "
    "hàm sigmoid và ngưỡng 0.5 để xác định nhãn dự đoán. Mô hình được lưu vào file model.pth sau khi huấn luyện. "
    "Đánh giá cuối cùng trên tập kiểm tra sử dụng độ chính xác tổng thể để đo lường hiệu suất."
)

# Thêm tiêu đề cấp 2: Kết quả triển khai
document.add_heading('Kết quả triển khai', level=2)
document.add_paragraph(
    "Quá trình huấn luyện cho thấy giá trị mất mát giảm đều qua các epoch, từ 75.5952 (epoch 1) xuống 8.7487 "
    "(epoch 5). Trên tập phát triển, mô hình đạt các chỉ số F1-score dao động từ 0.33 (Service) đến 0.93 "
    "(Public_area), với độ chính xác tổng thể (micro accuracy) là 89%. Trên tập kiểm tra, mô hình đạt độ "
    "chính xác tổng thể 96.19%, thể hiện khả năng tổng quát hóa tốt trên dữ liệu mới."
)

# Thêm tài liệu tham khảo
document.add_heading('Tài liệu tham khảo', level=2)
document.add_paragraph(
    "[1] Liu, Z., et al. (2022). A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on "
    "Computer Vision and Pattern Recognition (CVPR).",
    style='Body Text'
)

# Lưu tài liệu
document.save('ConvNeXt_Tiny_Methodology.docx')

print("Tài liệu Word đã được tạo và lưu tại 'ConvNeXt_Tiny_Methodology.docx'")