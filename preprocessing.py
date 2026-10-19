import os
import json
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
# ----- Biến config -----
IMG_DIR = "D:\\UIT\\NamBaUIT\\IE403\\image"
LABEL_FILE = "all_image_label.xlsx"
TRAIN_FILE = "train.json"
DEV_FILE = "dev.json"
TEST_FILE = "test.json"

# ----- Load label từ Excel -----
label_df = pd.read_excel(LABEL_FILE)

# Tạo label từ các cột Location, Food, Room, Facilities, Service, Public_area
label_columns = ["Location", "Food", "Room", "Facilities", "Service", "Public_area"]
print(f"Các cột trong file Excel: {list(label_df.columns)}")

# Hàm gộp nhãn: mỗi nhãn là một vector gồm các giá trị 0/1 tương ứng
def build_label(row):
    return [int(row[col]) if not pd.isna(row[col]) else 0 for col in label_columns]

# Tạo từ điển ánh xạ từ tên file sang nhãn vector
label_map = {
    row["file_name"]: build_label(row)
    for _, row in label_df.iterrows()
}

# ----- Hàm đọc json và tạo danh sách ảnh -----
def load_image_paths_and_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = []
    for item in data:
        # Lấy các tên file ảnh từ trường 'list_img', đây là một danh sách
        for filename in item.get("list_img", []):  # Duyệt qua các tên file ảnh trong 'list_img'
            label = label_map.get(filename)  # Lấy nhãn từ label_map với tên file ảnh
            if label is not None:
                samples.append((os.path.join(IMG_DIR, filename), label))  # Lưu đường dẫn và nhãn
    return samples

# ----- Custom Dataset -----
class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Chuyển label từ list thành tensor có dtype là torch.float32
        label = torch.tensor(label, dtype=torch.float32)  # Convert list to tensor
        
        return image, label

# ----- Transform -----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ----- Load dữ liệu -----
train_data = load_image_paths_and_labels(TRAIN_FILE)
dev_data = load_image_paths_and_labels(DEV_FILE)
test_data = load_image_paths_and_labels(TEST_FILE)

train_dataset = ImageDataset(train_data, transform=transform)
dev_dataset = ImageDataset(dev_data, transform=transform)
test_dataset = ImageDataset(test_data, transform=transform)

# ----- Kiểm tra ----- 
if __name__ == "__main__":
    print(f"Tổng số ảnh train: {len(train_dataset)}")
    print(f"Tổng số ảnh dev: {len(dev_dataset)}")
    print(f"Tổng số ảnh test: {len(test_dataset)}")
    print(train_dataset[0])  # In ra ảnh và nhãn của phần tử đầu tiên trong tập train
    
from collections import Counter

label_counter = Counter()
for _, label_tensor in train_dataset:
    for i, val in enumerate(label_tensor):
        if val == 1:
            label_counter[i] += 1

from preprocessing import label_columns
print("\n📊 Số lượng mẫu theo từng nhãn:")
for i, name in enumerate(label_columns):
    print(f"{name}: {label_counter[i]} mẫu")
    
# ---- Bổ sung vào cuối file preprocessing.py ----

from torch.utils.data import Dataset
from collections import Counter
import torch
import re


def extract_text_sentiment_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    for item in data:
        comment = item.get("comment", "").strip()
        text_labels = item.get("text_label", [])  

        sentiments = [lbl.split("#")[-1] for lbl in text_labels if "#" in lbl]

        # Gán nhãn cảm xúc chính cho đoạn văn bản
        if "Negative" in sentiments:
            sentiment_label = "Negative"
        elif "Positive" in sentiments:
            sentiment_label = "Positive"
        elif "Neutral" in sentiments:
            sentiment_label = "Neutral"
        else:
            sentiment_label = "Neutral"

        texts.append(comment)
        labels.append(sentiment_label)

    return texts, labels

# 1. Hàm tokenize đơn giản (chỉ tách từ theo khoảng trắng, loại bỏ ký tự đặc biệt)
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# 2. Load dữ liệu văn bản và label cảm xúc
texts_train, labels_train = extract_text_sentiment_data(TRAIN_FILE)
texts_dev, labels_dev = extract_text_sentiment_data(DEV_FILE)
texts_test, labels_test = extract_text_sentiment_data(TEST_FILE)

# 3. Xây vocab từ dữ liệu train
word_counter = Counter()
for text in texts_train:
    tokens = simple_tokenize(text)
    word_counter.update(tokens)

# Giới hạn vocab (nếu muốn), ví dụ: chỉ lấy 10_000 từ phổ biến nhất
MAX_VOCAB_SIZE = 10000
most_common = word_counter.most_common(MAX_VOCAB_SIZE)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}  # +2 để dành chỗ cho <unk> và <pad>
vocab["<pad>"] = 0
vocab["<unk>"] = 1

# 4. Ánh xạ label thành số
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}

# 5. Dataset cho văn bản
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        token_ids = token_ids[:self.max_len]  # Cắt bớt nếu quá dài
        padding = [self.vocab["<pad>"]] * (self.max_len - len(token_ids))
        input_tensor = torch.tensor(token_ids + padding, dtype=torch.long)
        label_tensor = torch.tensor(label_map[self.labels[idx]], dtype=torch.long)
        return input_tensor, label_tensor

# 6. Tạo các dataset
train_text_dataset = TextDataset(texts_train, labels_train, vocab, simple_tokenize)
dev_text_dataset = TextDataset(texts_dev, labels_dev, vocab, simple_tokenize)
test_text_dataset = TextDataset(texts_test, labels_test, vocab, simple_tokenize)

# 7. Test thử
if __name__ == "__main__":
    print("\n📝 Sample văn bản và nhãn:")
    print(train_text_dataset[0])