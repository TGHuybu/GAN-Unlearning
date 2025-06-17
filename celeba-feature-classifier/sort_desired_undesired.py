import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os
import shutil
import argparse
import numpy as np


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
        
        # # Load danh sách thuộc tính
        # with open(attributes_file, 'r') as f:
        #     self.attributes = f.readlines()[2].split()  # Lấy header từ file
        self.attributes = list(np.loadtxt(attributes_file, dtype=str))

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
    
    parser = argparse.ArgumentParser(description="Facial Attribute Classifier")
    parser.add_argument("inpath", type=str, help="path to the INPUT image folder")
    parser.add_argument("outpath", type=str, help="path to the OUTPUT image folder")
    parser.add_argument("model_path", type=str, help="path to the CLASSIFIER")
    parser.add_argument("attr_list", type=str, help="path to the ATTRIBUTE LIST")
    parser.add_argument(
        "--neg_class", 
        type=str, default="Eyeglasses", help="undesired attribute class (see file attr_names.txt)"
    )
    args = parser.parse_args()

    classifier = FacialAttributeClassifier(
        model_path=args.model_path,
        attributes_file=args.attr_list
    )
    
    # Đọc danh sách ảnh từ một thư mục hoặc file chứa đường dẫn ảnh
    image_folder = args.inpath
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
    
    # Thực hiện dự đoán trên tất cả các ảnh
    predictions = classifier.predict_batch(image_files)

    negative_attr = args.neg_class
    outfolder = args.outpath
    # os.makedirs(outfolder, exist_ok=True)
    os.makedirs(f"{outfolder}/{negative_attr}/data", exist_ok=True)
    os.makedirs(f"{outfolder}/no{negative_attr}/data", exist_ok=True)
    for pred in predictions:
        print(f"Image: {pred['image']}")
        if pred["attributes"][negative_attr]:
            dst = f"{outfolder}/{negative_attr}/data/{pred['image']}"
            shutil.copy(f"{image_folder}/{pred['image']}", dst)
        else:
            dst = f"{outfolder}/no{negative_attr}/data/{pred['image']}"
            shutil.copy(f"{image_folder}/{pred['image']}", dst)
