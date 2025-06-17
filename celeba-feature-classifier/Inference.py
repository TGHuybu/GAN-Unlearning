import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os

class FacialAttributeClassifier:
    def __init__(self, model_path, attributes_file):
        # Khởi tạo model
        self.model = models.resnext50_32x4d(pretrained=False)
        self.model.fc = nn.Linear(2048, 40)
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.eval()
        
        # Chuyển model sang GPU nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Chuẩn bị transform ảnh
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Load danh sách thuộc tính
        with open(attributes_file, 'r') as f:
            self.attributes = f.readlines()[2].split()  # Lấy header từ file

    def predict_batch(self, image_paths):
        results = []
        
        for image_path in image_paths:
            try:
                # Tiền xử lý ảnh
                img = Image.open(image_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # Dự đoán
                with torch.no_grad():
                    output = self.model(img_tensor)
                
                # Chuyển đổi kết quả
                predictions = (output >= 0).int().squeeze().cpu().numpy()
                result = {attr: bool(pred) for attr, pred in zip(self.attributes, predictions)}
                
                results.append({"image": os.path.basename(image_path), "attributes": result})
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results

# Cách sử dụng
if __name__ == "__main__":
    classifier = FacialAttributeClassifier(
        model_path="./models/model_2_epoch.pt",
        attributes_file="./CelebA/Anno/attr_names.txt"
    )
    
    # Đọc danh sách ảnh từ một thư mục hoặc file chứa đường dẫn ảnh
    image_folder = "/media02/lhthai/test/"
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
    
    # Thực hiện dự đoán trên tất cả các ảnh
    predictions = classifier.predict_batch(image_files)
    
    # Hiển thị kết quả cho từng ảnh
    for pred in predictions:
        print(f"Image: {pred['image']}")
        for attr, value in pred['attributes'].items():
            print(f"  {attr}: {'Có' if value else 'Không'}")
