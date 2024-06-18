# -*- coding: utf-8 -*-
"""
TILScout
@author: Huibo Zhang
"""

"""
### patch generation
"""
import openslide
import numpy as np
import tifffile as tiff
from openslide.deepzoom import DeepZoomGenerator
import os
import glob
import concurrent.futures
from normalize_HnE import norm_HnE
import logging

# Define constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
SIZE = 224  
BATCH_SIZE = 1000  
NUM_THREADS = 1

# Define results folder path
results_path = os.path.abspath("./results")
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Function to process images and generate patches
def process_image(directory_path, patch_path):
    label = os.path.basename(directory_path)
    print("Processing:", label)
    file_name = os.path.join(patch_path, label[:-4]) #label[:-4], exclude extension of original file
    os.makedirs(file_name, exist_ok=True)
    
    try:
        slide = openslide.OpenSlide(directory_path)
    except Exception as e:
        print(f"Error opening slide at {directory_path}: {str(e)}")
        return
    
    objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    tiles = DeepZoomGenerator(slide, tile_size=150, overlap=0, limit_bounds=False)
    level = tiles.level_count - 1 if objective < 40.0 else tiles.level_count - 2
    cols, rows = tiles.level_tiles[level]

    for row in range(rows):
        for col in range(cols):
            process_tile(tiles, level, col, row, file_name, label)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='tile_processing.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
def process_tile(tiles, level, col, row, file_name, label):
    tile_name = f"{col}_{row}"
    try:
        temp_tile = tiles.get_tile(level, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
            # Log processing event 
            norm_img, _, _ = norm_HnE(temp_tile_np, Io=240, alpha=1, beta=0.15)
            tiff.imsave(os.path.join(file_name, f"{tile_name}_{label[:23]}_norm.tif"), norm_img)
            logging.info(f"Processing tile number: {tile_name}")
        else:
            # Log skipping event
            logging.info(f"Skipping tile: {tile_name}")
    except Exception as e:
        # Log errors
        logging.error(f"Error processing tile {tile_name}: {str(e)}")

patch_path = os.path.abspath("./patch") #Create a temporary folder ./patch for storing patches
if not os.path.exists(patch_path):
    os.makedirs(patch_path)
directory_paths = glob.glob(os.path.join("WSI_example", "*.svs"))

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(process_image, path, patch_path) for path in directory_paths]
    concurrent.futures.wait(futures)



"""
### patch prediction and TIL score calculation
"""
import cv2
from tensorflow import keras
import pandas as pd
import gc
from concurrent.futures import ThreadPoolExecutor

SIZE = 224  
BATCH_SIZE = 1000  
NUM_THREADS = 1

# Load model
prediction_model = keras.models.load_model('best_InceptionResNetV2_model.h5')

def process_directory(directory_path):
    label = os.path.basename(directory_path)
    img_paths = glob.glob(os.path.join(directory_path, "*.tif"))
    num_images = len(img_paths)
    num_batches = np.ceil(num_images / BATCH_SIZE).astype(int)
    pred_ID, Y_proba = pd.DataFrame(), pd.DataFrame()

    for batch in range(num_batches):
        batch_img_paths = img_paths[batch * BATCH_SIZE: min((batch + 1) * BATCH_SIZE, num_images)]
        pred_images = [cv2.cvtColor(cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), (SIZE, SIZE)), cv2.COLOR_BGR2RGB) for img_path in batch_img_paths]
        X_pred = np.array(pred_images) / 255.0

        # Perform prediction
        y_proba = prediction_model.predict(X_pred)
        batch_pred_ID = pd.DataFrame(batch_img_paths, columns=['Patch_id'])
        batch_Y_proba = pd.DataFrame(y_proba, columns=['Positive', 'Negative', 'Other'])

        pred_ID = pd.concat([pred_ID, batch_pred_ID], ignore_index=True)
        Y_proba = pd.concat([Y_proba, batch_Y_proba], ignore_index=True)

        del pred_images, X_pred
        gc.collect()

    # Combine results
    results = pd.concat([pred_ID, Y_proba], axis=1)
    results['MaxCategory'] = results[['Positive', 'Negative', 'Other']].idxmax(axis=1)
    results['Coordinates'] = results['Patch_id'].str.extract(r'(\d+_\d+)')
    results[['X', 'Y']] = results['Coordinates'].str.split('_', expand=True).astype(int)
    
    # Calculate TIL scores
    TIL_positive = results[results["Positive"] > results[["Negative", "Other"]].max(axis=1)]
    TIL_negative = results[results["Negative"] > results[["Positive", "Other"]].max(axis=1)]
    #Other = results[results["Other"] > results[["Positive", "Negative"]].max(axis=1)]

    score = TIL_positive.shape[0] / (TIL_negative.shape[0] + TIL_positive.shape[0])
    result_text = f"{label}: TIL Score = {score:.4f}"
    print(result_text)
    with open(os.path.join(results_path, f"{label}_TIL_score.txt"), 'w') as f:
        print(label, score, file=f)
    return results

# Processing all directories
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    directory_paths = glob.glob(os.path.join("patch", "*"))  
    results_list = list(executor.map(process_directory, directory_paths))



"""
### TIL map generation
"""
import matplotlib.pyplot as plt

def generate_til_maps(results_list):
    for results in results_list:
        draw_til_map(results)

def draw_til_map(results):
    # Extraction of the directory name
    first_path = results['Patch_id'].iloc[0]
    label1 = os.path.basename(os.path.dirname(first_path))
    #print("Generating TIL map for:", label1)
    colors = {'Negative': 'red', 'Positive': 'purple', 'Other': 'pink'}
    max_dims = results[['X', 'Y']].max()
    plt.figure(figsize=(max_dims['X']/7.1, max_dims['Y']/7.1))  # Adjust figure size dynamically

    for category, color in colors.items():
        category_data = results[results['MaxCategory'] == category]
        plt.scatter(x=category_data.X, y=category_data.Y, s=48, color=color, marker='s')

    plt.title(f"{label1} TIL Map", fontsize=max_dims['X']/7.1)  # Title size adjusts with figure size
    plt.gca().set_aspect('equal')
    plt.xlim(0, max_dims['X'] + 5)
    plt.ylim(max_dims['Y'] + 5, 0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{results_path}/{label1}_TIL_map.pdf")
    plt.close()
    print("TIL map is generated and stored in the results directory for:", label1)
generate_til_maps(results_list)

#delete temporary folder patch
import shutil
patch_path = os.path.abspath("./patch")
shutil.rmtree(patch_path)

