import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from transformers import AutoTokenizer, AutoModel

# -------------------------------
# 📁 Định nghĩa Dataset
# -------------------------------
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

# -------------------------------
# 🧠 Bộ Trích Xuất Đặc Trưng Văn Bản
# -------------------------------
class TextFeatureExtractor(nn.Module):
    def __init__(self, model_name="vinai/phobert-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size

    def forward(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        input_ids = encoded_input['input_ids'].to(self.model.device)
        attention_mask = encoded_input['attention_mask'].to(self.model.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding.float()

# -------------------------------
# 🧩 Mô hình Phân Tích Cảm Xúc Đa Phương Thức
# -------------------------------
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
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    print("Bắt đầu huấn luyện mô hình...")
    for epoch in range(epochs):
        total_loss = 0
        for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    print("Huấn luyện hoàn tất.")

def evaluate_and_report(model, dataloader, label_list, device, eval_set_name="Development"):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\nBắt đầu đánh giá chi tiết trên tập {eval_set_name}...")
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

    print(f"\n📊 Báo cáo phân loại trên tập {eval_set_name}:")
    print(classification_report(y_true, y_pred, target_names=label_list, zero_division=0))
    print(f"Đánh giá chi tiết trên tập {eval_set_name} hoàn tất.")

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
if __name__ == '__main__':
    # --- Cấu hình ---
    IMAGE_DIR = "D:\\UIT\\NamBaUIT\\IE403\\image"
    TRAIN_JSON_PATH = "train.json"
    DEV_JSON_PATH = "dev.json"
    TEST_JSON_PATH = "test.json"
    MODEL_SAVE_PATH = "multimodal_model.pth"

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # --- Tải và xử lý dữ liệu ---
    def load_json_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    train_data = load_json_data(TRAIN_JSON_PATH)
    dev_data = load_json_data(DEV_JSON_PATH)
    test_data = load_json_data(TEST_JSON_PATH)

    all_labels = set()
    for entry in train_data + dev_data + test_data:
        for label in entry.get('text_img_label', []):
            all_labels.add(label)

    label_columns = sorted(list(all_labels))
    num_classes = len(label_columns)

    print(f"Tổng số nhãn Aspect#Sentiment (classes): {num_classes}")
    print(f"Danh sách nhãn: {label_columns}")

    label_to_idx = {label: i for i, label in enumerate(label_columns)}
    idx_to_label = {i: label for i, label in enumerate(label_columns)}

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

    print(f"Số lượng mẫu train: {len(train_dataset)}")
    print(f"Số lượng mẫu dev: {len(dev_dataset)}")
    print(f"Số lượng mẫu test: {len(test_dataset)}")

    model = MultimodalSentimentModel(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- Bắt đầu quá trình huấn luyện ---")
    train_model(model, train_loader, criterion, optimizer, device, EPOCHS)
    print("--- Kết thúc quá trình huấn luyện ---")

    evaluate_and_report(model, dev_loader, label_columns, device, eval_set_name="Development")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nMô hình đã được lưu tại: {MODEL_SAVE_PATH}")

    acc_test = evaluate_accuracy(model, test_loader, device, eval_set_name="Test")
    print(f"Độ chính xác trên tập test (micro accuracy): {acc_test*100:.2f}%")


