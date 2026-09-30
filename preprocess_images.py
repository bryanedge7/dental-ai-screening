import cv2
import os
import numpy as np

def apply_clahe_and_scale():
    image_dir = os.path.join('data', 'processed', 'images', 'train')
    if not os.path.exists(image_dir):
        print(f"Error: Target directory {image_dir} not found!")
        return

    # Initialize clinical CLAHE filter
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Define a standard uniform resolution for the dental screening model
    TARGET_WIDTH = 1024
    TARGET_HEIGHT = 512
    
    print("Beginning professional CLAHE and uniform scaling pipeline...")
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
    count = 0
    for filename in images:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # 1. Apply contrast enhancement filter
            enhanced_img = clahe.apply(img)
            
            # 2. Scale image to uniform clinical resolution
            standardized_img = cv2.resize(enhanced_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # Overwrite original with the optimized version
            cv2.imwrite(img_path, standardized_img)
            count += 1
            if count % 100 == 0:
                print(f"Successfully optimized and standardized {count}/{len(images)} X-ray scans...")

    print(f"🎉 Complete! Successfully enhanced and scaled all {count} panoramic X-rays to uniform {TARGET_WIDTH}x{TARGET_HEIGHT} resolution.")

if __name__ == '__main__':
    apply_clahe_and_scale()