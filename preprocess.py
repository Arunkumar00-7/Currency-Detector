import os
import cv2
import numpy as np

# Input dataset folders
data_dirs = {
    "100": "D:/vsproject/Currency_Detector_Website/dataset/100 rs note",
    "200": "D:/vsproject/Currency_Detector_Website/dataset/200 rs note",
    "500": "D:/vsproject/Currency_Detector_Website/dataset/500 rs note"
}

# Output folder
output_dir = "D:/Miniproject/bolt/project/processed_dataset"
os.makedirs(output_dir, exist_ok=True)

def preprocess_images():
    for label, folder in data_dirs.items():
        save_path = os.path.join(output_dir, label)
        os.makedirs(save_path, exist_ok=True)
        
        for i, filename in enumerate(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping invalid image: {img_path}")
                continue
            
            # Resize to 224x224
            img = cv2.resize(img, (224, 224))
            
            # Normalize (convert pixel values to range 0-1)
            img = img / 255.0
            
            # Save preprocessed image
            new_filename = f"{label}_{i}.jpg"
            cv2.imwrite(os.path.join(save_path, new_filename), (img * 255).astype(np.uint8))
            
            print(f"Processed: {new_filename}")

if __name__ == "__main__":
    preprocess_images()
    print("✅ Image preprocessing complete! Check the processed_dataset folder.")
