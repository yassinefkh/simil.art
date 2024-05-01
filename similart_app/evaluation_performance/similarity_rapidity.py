
from django.conf import settings
import time
import random
import numpy as np
import os
import csv
from cnn_model.similarity import similarity
from cnn_model.features_extraction import get_vector
from indexing.faiss_indexing import load_faiss_index, search_similar_images_faiss
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn






# Ask the user for the layer to test
layer = input("Enter the layer to test (layer4, layer3, layer2, avgpool): ")
dict_agvpool = torch.load(settings.FEATURES_VECTORS_PATH_AVGPOOL)
DATABASE_DIRECTORY = os.path.join(settings.MEDIA_ROOT, 'wikiartresized')



# Path to the FAISS index file
if layer == "layer4":
    FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH_LAYER4
    JOBLIB_PATH = settings.PCA_PATH_LAYER4
elif layer == "layer3":
    FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH_LAYER3
    JOBLIB_PATH = settings.PCA_PATH_LAYER3
elif layer == "layer2":
    FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH_LAYER2
    JOBLIB_PATH = settings.PCA_PATH_LAYER2
else:
    FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH_AVGPOOL 
    JOBLIB_PATH = None
    
    


# List of database images
database_images = []
for root, _, files in os.walk(DATABASE_DIRECTORY):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            database_images.append(os.path.join(root, file))
            
            

# List of query images
query_images = [
    '/home/yassfkh/Desktop/L3H1/2023-l3h1/branches/testYassine/WebApp/ProjetCBIR/core_interface/media/wikiartresized/Art_Nouveau_Modern/a.y.-jackson_first-snow-algoma-country-1920.jpg',
]



# Number of times to repeat the experiment
N_REPEATS = 5


# Open the CSV file for writing results
csv_file = open('results.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Number of Images', 'Average Time Method 1 (similarity())', 'Average Time Method 2 (with FAISS)'])



# List of database sizes to test
SIZES = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]

for size in SIZES:
    # Sample the database images
    database_images_sample = random.sample(database_images, size)

    # Method 1: similarity()
    times1 = []
    for _ in range(N_REPEATS):
        for query_image in query_images:
            for database_image in database_images_sample:
                tic = time.perf_counter()
                similarities = [] 
                if layer == "avgpool":  
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    pic_vector = get_vector(query_image, f'RESNET18_{layer.upper()}')
                    for file, output_vector in dict_agvpool.items():
                        cos_sim = cos(pic_vector.unsqueeze(0), output_vector.unsqueeze(0))
                        similarities.append(cos_sim.item())
                else:
                    pca = load(JOBLIB_PATH)
                    pic_vector = get_vector(query_image, f'RESNET18_{layer.upper()}')
                    pic_vector_pca = pca.transform(pic_vector.numpy().flatten().reshape(1, -1))
                    dict_layer = torch.load(os.path.join(settings.BASE_DIR, f'resources/vectors/output_dict_wikiart_pca_{layer}.pt'))
                    for file, output_vector in dict_layer.items():
                        cos_sim = cosine_similarity(pic_vector_pca, output_vector.reshape(1, -1))
                        similarities.append(cos_sim[0][0])

                toc = time.perf_counter()
                times1.append(toc - tic)

    # Method 2: search_similar_images_faiss()
    times2 = []
    for _ in range(N_REPEATS):
        for query_image in query_images:
            faiss_index = load_faiss_index(FAISS_INDEX_PATH)
            vector = get_vector(query_image, f'RESNET18_{layer.upper()}')
            vector = vector.numpy().reshape(1, -1).astype('float32')
            for database_image in database_images_sample:
                tic = time.perf_counter()
                search_similar_images_faiss(vector, faiss_index, 1000)
                toc = time.perf_counter()
                times2.append(toc - tic)

    # Calculate average times
    avg_time1 = np.mean(times1)
    avg_time2 = np.mean(times2)

    print(f"Method 1 (similarity()) : Average execution time per trial = {avg_time1:.4f} seconds")
    print(f"Method 2 (with FAISS) : Average execution time per trial = {avg_time2:.4f} seconds")

    csv_writer.writerow([size, avg_time1, avg_time2])

csv_file.close()
