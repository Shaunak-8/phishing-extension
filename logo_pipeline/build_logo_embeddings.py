#!/usr/bin/env python3
"""
build_logo_embeddings.py

Load logos from a directory (logos/<brand>/*.png) and compute embeddings.
Primary method: use TensorFlow MobileNetV2 for embeddings (if installed).
Fallback: simple resized+flattened RGB normalized vector (fast, lightweight).
Outputs a JSON file with brand -> embedding vector and an .npy array for quick loads.

Usage:
  python build_logo_embeddings.py --logos logos --out logo_db_embeddings.json
"""
import argparse
import os
import json
import numpy as np
from PIL import Image

def tf_embedding(img, model, input_size=(224, 224)):
    """
    Compute embedding using TensorFlow MobileNetV2.

    Args:
        img (PIL Image): Input image.
        model (TensorFlow Model): MobileNetV2 model.
        input_size (tuple): Input size for the model.

    Returns:
        list: Embedding vector.
    """
    img = img.resize(input_size).convert('RGB')
    arr = np.array(img).astype('float32') / 127.5 - 1.0
    x = np.expand_dims(arr, axis=0)
    emb = model.predict(x)
    emb = emb.flatten().tolist()
    return emb

def simple_embedding(img, size=(128, 128)):
    """
    Compute simple embedding by resizing and normalizing the image.

    Args:
        img (PIL Image): Input image.
        size (tuple): Output size.

    Returns:
        list: Embedding vector.
    """
    img = img.resize(size).convert('RGB')
    arr = np.array(img).astype('float32') / 255.0
    mean = arr.mean(axis=(0, 1)).tolist()
    std = arr.std(axis=(0, 1)).tolist()
    reduced = np.array(Image.fromarray((arr * 255).astype('uint8')).resize((16, 16))).astype('float32') / 255.0
    vec = reduced.flatten().tolist()
    return mean + std + vec

def main():
    """
    Main function to compute logo embeddings.
    """
    parser = argparse.ArgumentParser(description='Compute logo embeddings')
    parser.add_argument('--logos', default='logos', help='logos/<brand> folders')
    parser.add_argument('--out', default='logo_db_embeddings.json')
    args = parser.parse_args()

    use_tf = False
    model = None
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.models import Model
        base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        model = base
        use_tf = True
        print("Using TensorFlow MobileNetV2 for embeddings (pooling=avg).")
    except Exception as e:
        print("TensorFlow not available or failed to load. Falling back to simple embedding. Error:", e)

    results = {}
    if not os.path.exists(args.logos):
        print("Logos folder not found:", args.logos)
        return

    for brand in sorted(os.listdir(args.logos)):
        bpath = os.path.join(args.logos, brand)
        if not os.path.isdir(bpath):
            continue
        embeddings = []
        for fn in os.listdir(bpath):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.ico', '.svg')):
                continue
            fpath = os.path.join(bpath, fn)
            try:
                img = Image.open(fpath).convert('RGB')
                if use_tf and model is not None:
                    emb = tf_embedding(img, model)
                else:
                    emb = simple_embedding(img)
                embeddings.append(emb)
            except Exception as e:
                print("Error processing", fpath, e)
        if embeddings:
            avg = np.mean(np.array(embeddings), axis=0).tolist()
            results[brand] = {'embedding': avg, 'count': len(embeddings)}
            print("Brand", brand, "=>", len(embeddings), "images")
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f)
    print("Saved embeddings to", args.out)

if __name__ == '__main__':
    main()