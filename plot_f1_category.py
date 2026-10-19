import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dữ liệu F1-score theo các chủ đề
data = {
    'Category': ['Facilities', 'Food', 'Location', 'Public_area', 'Room', 'Service'],
    'ConvNeXt_Tiny': [0.20, 0.38, 0.70, 0.73, 0.80, 0.87],
    'FCMF': [0.67, 0.88, 0.95, 0.89, 0.92, 0.96],
    'FLAVA': [0.70, 0.77, 0.81, 0.91, 0.92, 0.94],
    'ViLT': [0.67, 0.69, 0.65, 0.94, 0.92, 0.86]
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
ax.set_title('F1-score theo từng nhóm chủ đề cho 4 mô hình')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.05)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()
