#!/usr/bin/env python3
"""
Resize Digit Images Script
Resizes all digit images from 128x128 to 28x28 for digit recognition training.
Run this script from the digit_data_base folder.
"""

import cv2
import os
import numpy as np
from pathlib import Path
import shutil

def resize_digit_images(create_backup=True, target_size=(28, 28)):
    """
    Resize all digit images in num_* folders from 128x128 to 28x28.
    
    Args:
        create_backup: Whether to backup original images
        target_size: Target size for resized images (width, height)
    """
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"Processing digit images in: {current_dir}")
    
    # Find all num_* folders
    digit_folders = [f for f in current_dir.iterdir() 
                    if f.is_dir() and f.name.startswith('num_')]
    
    if not digit_folders:
        print("❌ No num_* folders found! Make sure you're running this from the digit_data_base folder.")
        return
    
    digit_folders.sort()  # Sort for consistent processing order
    print(f"Found {len(digit_folders)} digit folders: {[f.name for f in digit_folders]}")
    
    # Create backup if requested
    if create_backup:
        backup_dir = current_dir / "original_128x128_backup"
        if not backup_dir.exists():
            print(f"Creating backup at: {backup_dir}")
            backup_dir.mkdir()
            
            # Copy all num_* folders to backup
            for folder in digit_folders:
                shutil.copytree(folder, backup_dir / folder.name)
            print("✅ Backup created successfully!")
    
    # Process each digit folder
    total_processed = 0
    
    for digit_folder in digit_folders:
        print(f"\n📂 Processing folder: {digit_folder.name}")
        
        # Find all image files
        image_files = list(digit_folder.glob("*.png"))
        image_files.extend(digit_folder.glob("*.jpg"))
        image_files.extend(digit_folder.glob("*.jpeg"))
        
        if not image_files:
            print(f"   ⚠️  No image files found in {digit_folder.name}")
            continue
        
        print(f"   Found {len(image_files)} images")
        
        # Process each image
        processed_count = 0
        failed_count = 0
        
        for img_file in image_files:
            try:
                # Load image
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"   ❌ Failed to load: {img_file.name}")
                    failed_count += 1
                    continue
                
                # Check original size
                original_shape = img.shape
                
                # Apply slight Gaussian blur to reduce aliasing when downscaling
                if original_shape[0] > target_size[0] * 2:  # Only blur if significant downscaling
                    img = cv2.GaussianBlur(img, (3, 3), 0.5)
                
                # Resize using INTER_AREA (best for downscaling)
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # Save resized image (overwrite original)
                success = cv2.imwrite(str(img_file), resized)
                
                if success:
                    processed_count += 1
                else:
                    print(f"   ❌ Failed to save: {img_file.name}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"   ❌ Error processing {img_file.name}: {e}")
                failed_count += 1
        
        print(f"   ✅ Processed: {processed_count}, Failed: {failed_count}")
        total_processed += processed_count
    
    print(f"\n🎉 Resize complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Target size: {target_size[0]}x{target_size[1]} pixels")
    
    if create_backup:
        print(f"Original images backed up to: original_128x128_backup/")
    
    # Verify a few resized images
    print(f"\n🔍 Verifying resize results...")
    sample_folder = digit_folders[0]
    sample_images = list(sample_folder.glob("*.png"))[:3]
    
    for sample_img in sample_images:
        img = cv2.imread(str(sample_img), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"   {sample_img.name}: {img.shape} ✅")
        else:
            print(f"   {sample_img.name}: Failed to load ❌")


def show_sample_images(num_samples=5):
    """Show sample resized images for verification."""
    try:
        import matplotlib.pyplot as plt
        
        current_dir = Path.cwd()
        digit_folders = [f for f in current_dir.iterdir() 
                        if f.is_dir() and f.name.startswith('num_')]
        
        if not digit_folders:
            print("No digit folders found for sampling")
            return
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.flatten()
        
        for i, digit_folder in enumerate(digit_folders[:10]):  # Show up to 10 digits
            image_files = list(digit_folder.glob("*.png"))
            
            if image_files:
                # Load first image from this digit folder
                img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    axes[i].imshow(img, cmap='gray')
                    axes[i].set_title(f'Digit {digit_folder.name[-1]}\n{img.shape}')
                    axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Sample Resized Digit Images', y=1.02)
        plt.show()
        
    except ImportError:
        print("matplotlib not available for preview")


if __name__ == "__main__":
    print("🔄 Digit Image Resizer")
    print("=" * 50)
    
    # Ask user for confirmation
    response = input("This will resize all images from 128x128 to 28x28.\nCreate backup first? [Y/n]: ")
    create_backup = response.lower() not in ['n', 'no']
    
    # Run resize
    resize_digit_images(create_backup=create_backup)
    
    # Ask if user wants to see samples
    response = input("\nShow sample resized images? [y/N]: ")
    if response.lower() in ['y', 'yes']:
        show_sample_images()
    
    print("\n✅ Done! Your digit images are now ready for training.")
