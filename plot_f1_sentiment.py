import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dữ liệu F1-score theo các chủ đề
data = {
    'Category': ['Negative', 'Neutral', 'Positive'],
    'ConvNeXt_Tiny': [0.55, 0.04, 0.55],
    'FCMF': [0.46, 0.13, 0.81],
    'FLAVA': [0.34, 0.25, 0.62],
    'ViLT': [0.40, 0.29, 0.74]
}

# Chuyển thành DataFrame
df = pd.DataFrame(data)

# Cài đặt cho biểu đồ
categories = df['Category']
x = np.arange(len(categories))  # nhãn trục x
width = 0.2  # độ rộng mỗi cột

# Tạo biểu đồ
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, df['ConvNeXt_Tiny'], width, label='ConvNeXt_Tiny')
ax.bar(x - 0.5*width, df['FCMF'], width, label='FCMF')
ax.bar(x + 0.5*width, df['FLAVA'], width, label='FLAVA')
ax.bar(x + 1.5*width, df['ViLT'], width, label='ViLT')

# Tuỳ chỉnh hiển thị
ax.set_xlabel('Category')
ax.set_ylabel('F1-score')
ax.set_title('F1-score theo từng nhóm cảm xúc cho 4 mô hình')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.05)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()
