import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import json
import os
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# --------------------------------------------------------------------------------
# ĐỊNH NGHĨA CÁC HÀM VÀ CLASS (Ở PHẠM VI TOÀN CỤC)
# --------------------------------------------------------------------------------

# Hàm helper để tải dữ liệu JSON
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Class Custom Dataset
class MultimodalDataset(Dataset):
    def __init__(self, data_list, image_dir, label_to_idx, transform=None):
        self.data_list = data_list
        self.image_dir = image_dir
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.num_classes = len(label_to_idx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        comment = entry.get('comment', "")
        img_file_name = entry.get('list_img', [None])[0]
        image_path = os.path.join(self.image_dir, img_file_name) if img_file_name else None

        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.zeros(3, 224, 224)

        labels_one_hot = torch.zeros(self.num_classes)
        for label_str in entry.get('text_img_label', []):
            if label_str in self.label_to_idx:
                labels_one_hot[self.label_to_idx[label_str]] = 1

        return image, comment, labels_one_hot

# Class Text Feature Extractor
class TextFeatureExtractor(nn.Module):
    def __init__(self, phobert_model_name="vinai/phobert-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(phobert_model_name)
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        self.output_dim = self.phobert.config.hidden_size

    def forward(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        input_ids = encoded_input['input_ids'].to(self.phobert.device)
        attention_mask = encoded_input['attention_mask'].to(self.phobert.device)

        with torch.no_grad():
            output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding.float()

# Class Multimodal Sentiment Model
class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        convnext_model = convnext_tiny(weights=weights)
        self.image_encoder = nn.Sequential(
            convnext_model.features,
            convnext_model.avgpool,
            nn.Flatten()
        )
        self.image_feature_dim = 768

        self.text_encoder = TextFeatureExtractor()
        self.text_feature_dim = self.text_encoder.output_dim

        combined_feature_dim = self.image_feature_dim + self.text_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        combined_features = torch.cat((image_features, text_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

# Hàm Huấn luyện

def train_model(model, train_loader, criterion, optimizer, device, EPOCHS):
    model.train()
    print("Bắt đầu huấn luyện mô hình...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
    print("Huấn luyện hoàn tất.")

# Hàm Đánh giá chi tiết

def evaluate_and_report(model, dataloader, label_list, device, eval_set_name="Evaluation"):
    model.eval()
    all_preds = []
    all_labels = []

    idx_to_label = {i: label for i, label in enumerate(label_list)}

    print(f"\n📌 Bắt đầu đánh giá trên tập {eval_set_name}...")

    with torch.no_grad():
        for images, texts, labels in tqdm(dataloader, desc=f"Evaluating {eval_set_name}"):
            images = images.float().to(device)
            labels = labels.float().to(device)

            outputs = model(images, texts)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()

            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    print(f"\n🔎 [Aspect#Sentiment] Classification Report ({eval_set_name}):")
    print(classification_report(y_true, y_pred, target_names=label_list, zero_division=0))

    aspect_true, sentiment_true = [], []
    aspect_pred, sentiment_pred = [], []

    for i in range(len(y_true)):
        true_aspects, pred_aspects = [], []
        true_sentiments, pred_sentiments = [], []

        for j in range(len(label_list)):
            if y_true[i][j] == 1:
                aspect, sentiment = label_list[j].split("#")
                true_aspects.append(aspect)
                true_sentiments.append(sentiment)
            if y_pred[i][j] == 1:
                aspect, sentiment = label_list[j].split("#")
                pred_aspects.append(aspect)
                pred_sentiments.append(sentiment)

        aspect_true.append(true_aspects)
        sentiment_true.append(true_sentiments)
        aspect_pred.append(pred_aspects)
        sentiment_pred.append(pred_sentiments)

    aspect_set = sorted(list(set([a for sublist in aspect_true + aspect_pred for a in sublist])))
    sentiment_set = sorted(list(set([s for sublist in sentiment_true + sentiment_pred for s in sublist])))

    mlb_aspect = MultiLabelBinarizer(classes=aspect_set)
    mlb_sentiment = MultiLabelBinarizer(classes=sentiment_set)

    y_true_aspect_bin = mlb_aspect.fit_transform(aspect_true)
    y_pred_aspect_bin = mlb_aspect.transform(aspect_pred)

    y_true_sentiment_bin = mlb_sentiment.fit_transform(sentiment_true)
    y_pred_sentiment_bin = mlb_sentiment.transform(sentiment_pred)

    print(f"\n🔎 [Aspect-only] Classification Report ({eval_set_name}):")
    print(classification_report(y_true_aspect_bin, y_pred_aspect_bin, target_names=mlb_aspect.classes_, zero_division=0))

    print(f"\n🔎 [Sentiment-only] Classification Report ({eval_set_name}):")
    print(classification_report(y_true_sentiment_bin, y_pred_sentiment_bin, target_names=mlb_sentiment.classes_, zero_division=0))

    print(f"✅ Đánh giá chi tiết trên tập {eval_set_name} hoàn tất.\n")

# Hàm Đánh giá micro accuracy

def evaluate_accuracy(model, loader, device, eval_set_name="Test"):
    model.eval()
    total_correct_predictions = 0
    total_num_elements = 0

    print(f"\nĐánh giá micro accuracy trên tập {eval_set_name}...")
    with torch.no_grad():
        for images, texts, labels in tqdm(loader, desc=f"Calculating micro accuracy on {eval_set_name}"):
            images = images.float().to(device)
            labels = labels.float().to(device)

            outputs = model(images, texts)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            total_correct_predictions += (predicted == labels).sum().item()
            total_num_elements += torch.numel(labels)

    if total_num_elements == 0:
        return 0.0
    return total_correct_predictions / total_num_elements

# --------------------------------------------------------------------------------
# MAIN CHƯƠNG TRÌNH CHÍNH
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    IMAGE_DIR = "D:\\UIT\\NamBaUIT\\IE403\\image"
    TRAIN_JSON_PATH = "train.json"
    DEV_JSON_PATH = "dev.json"
    TEST_JSON_PATH = "test.json"
    MODEL_SAVE_PATH = "multimodal_model.pth"

    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4
    EPOCHS = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    train_data = load_json_data(TRAIN_JSON_PATH)
    dev_data = load_json_data(DEV_JSON_PATH)
    test_data = load_json_data(TEST_JSON_PATH)

    all_aspect_sentiment_labels = set()
    for entry in train_data + dev_data + test_data:
        for label in entry.get('text_img_label', []):
            all_aspect_sentiment_labels.add(label)

    label_columns = sorted(list(all_aspect_sentiment_labels))
    num_classes = len(label_columns)

    print(f"Tổng số nhãn Aspect#Sentiment (classes): {num_classes}")
    print(f"Danh sách nhãn: {label_columns}")

    label_to_idx = {label: i for i, label in enumerate(label_columns)}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultimodalDataset(train_data, IMAGE_DIR, label_to_idx, transform)
    dev_dataset = MultimodalDataset(dev_data, IMAGE_DIR, label_to_idx, transform)
    test_dataset = MultimodalDataset(test_data, IMAGE_DIR, label_to_idx, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultimodalSentimentModel(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- Bắt đầu quá trình huấn luyện ---")
    train_model(model, train_loader, criterion, optimizer, device, EPOCHS)
    print("--- Kết thúc quá trình huấn luyện ---")

    evaluate_and_report(model, train_loader, label_columns, device, eval_set_name="Train")
    evaluate_and_report(model, dev_loader, label_columns, device, eval_set_name="Dev")
    evaluate_and_report(model, test_loader, label_columns, device, eval_set_name="Test")

    acc_test = evaluate_accuracy(model, test_loader, device, eval_set_name="Test")
    print(f"\n🎯 Micro Accuracy trên tập test: {acc_test * 100:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 Mô hình đã được lưu tại: {MODEL_SAVE_PATH}")
