import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from tqdm import tqdm
from pathlib import Path

def apply_ripening_transformation(img, maturity_level):
    """
    Transform dark green fruits to orange with 4 maturity levels:
    0 = original dark green (immature)
    1 = yellowish-green
    2 = light orange
    3 = fully ripe orange
    """
    # Convert to HSV for color manipulation
    img = img.convert('RGB')
    hsv = img.convert('HSV')
    h, s, v = hsv.split()
    
    # Convert to numpy arrays
    h = np.array(h, dtype=np.float32)
    s = np.array(s, dtype=np.float32)
    v = np.array(v, dtype=np.float32)
    
    # Define target hues for each maturity level
    hue_targets = [None, 60, 30, 20]  # None=keep original, 60=yellowish, 30=orange, 20=deep orange
    
    if maturity_level > 0:
        # Calculate hue shift - stronger transformation for higher levels
        if maturity_level == 1:
            # Shift green (typically 80-120) towards yellow (60)
            h = np.where(h > 60, h * 0.7 + 18, h)  # More aggressive shift for greens
        else:
            # For orange stages, shift directly to target hue
            target_hue = hue_targets[maturity_level]
            blend_factor = 0.8 if maturity_level == 3 else 0.6  # Stronger blend for final stage
            h = (1 - blend_factor) * h + blend_factor * target_hue
        
        # Adjust saturation - increase for ripe fruits
        s = s * (1 + 0.4 * maturity_level)
        
        # Adjust brightness - ripe fruits are slightly brighter
        v = v * (1 + 0.15 * maturity_level)
    
    # Clamp values
    h = np.clip(h, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Images
    h = Image.fromarray(h)
    s = Image.fromarray(s)
    v = Image.fromarray(v)
    
    # Merge back to RGB
    ripe_hsv = Image.merge('HSV', (h, s, v))
    ripe_rgb = ripe_hsv.convert('RGB')
    
    # Additional effects for ripe stages
    if maturity_level >= 2:
        # Add glow effect
        glow = ripe_rgb.filter(ImageFilter.GaussianBlur(radius=2))
        ripe_rgb = Image.blend(ripe_rgb, glow, alpha=0.15)
        
        if maturity_level == 3:
            # Add fruit texture and highlights
            texture = np.random.normal(128, 10, ripe_rgb.size[::-1]).astype(np.uint8)
            texture = Image.fromarray(texture).convert('L')
            ripe_rgb = Image.composite(
                ripe_rgb,
                ImageEnhance.Brightness(ripe_rgb).enhance(1.1),
                texture
            )
            # Enhance contrast
            ripe_rgb = ImageEnhance.Contrast(ripe_rgb).enhance(1.1)
    
    return ripe_rgb

def process_crops(input_dir, output_dir):
    """Process all crops from class_1 into 4 maturity classes"""
    classes = [
        '1_dark_green', 
        '2_yellowish_green', 
        '3_light_orange', 
        '4_ripe_orange'
    ]
    
    # Create output directories
    for cls in classes:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
    
    # Create label file
    with open(output_dir / 'maturity_labels.csv', 'w') as f:
        f.write("filename,class,class_name,original_file\n")
    
    # Get all crop images from class_1
    crop_files = list((input_dir / "class_1").glob('*.[jJpP][pPnN][gG]'))
    random.shuffle(crop_files)
    
    # Process each crop
    for i, crop_file in enumerate(tqdm(crop_files, desc="Creating maturity classes")):
        try:
            with Image.open(crop_file) as img:
                maturity_class = i % 4
                class_name = classes[maturity_class]
                
                # Apply transformation
                transformed_img = apply_ripening_transformation(img, maturity_class)
                
                # Save with new filename
                new_filename = f"{maturity_class+1}_{crop_file.stem}.jpg"
                output_path = output_dir / class_name / new_filename
                transformed_img.save(output_path, quality=95)
                
                # Record metadata
                with open(output_dir / 'maturity_labels.csv', 'a') as f:
                    f.write(f"{output_path},{maturity_class},{class_name},{crop_file.name}\n")
        except Exception as e:
            print(f"Skipping {crop_file.name}: {str(e)}")
            continue

if __name__ == "__main__":
    # Use class_1 directory as input
    input_dir = Path("/home/nery/desafio_agscan/ds_terrestres_results")
    output_dir = Path("/home/nery/desafio_agscan/maturity_classes_final")
    
    print("Creating orange ripening transformation...")
    process_crops(input_dir, output_dir)
    
    print(f"\nDone! Created 4 maturity classes in: {output_dir}")
    print("Ripening progression:")
    print("1. Dark green (immature)")
    print("2. Yellowish-green")
    print("3. Light orange")
    print("4. Ripe orange")
