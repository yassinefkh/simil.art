
from django.conf import settings

import os
import json
import time

import numpy as np
import faiss
import h5py
from joblib import load
from sklearn.decomposition import PCA
import torch

from cnn_model.features_extraction import get_vector


def load_faiss_index(index_path):
    """
    Load a FAISS index from disk.

    Args:
    - index_path: Path to the FAISS index file.

    Returns:
    - A faiss.Index object representing the loaded index or None on failure.
    """
    try:
        return faiss.read_index(index_path)
    except FileNotFoundError:
        print("L'index FAISS spécifié est introuvable.")
    except Exception as e:
        print(f"Erreur lors du chargement de l'index FAISS : {e}")
        
        
def create_faiss_index(feature_vectors, use_quantization=False):
    """
    Create and return a FAISS FLATL2 index from feature vectors.

    Args:
    - feature_vectors: Numpy array of feature vectors.
    - use_quantization: Boolean to enable or disable quantization.

    Returns:
    - A faiss Index object representing the created index.
    """
    dimension = feature_vectors.shape[1]
    if use_quantization:
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        if not index.is_trained:
            index.train(feature_vectors)
    else:
        index = faiss.IndexFlatL2(dimension)
    index.add(feature_vectors)
    return index


def search_similar_images_faiss(query_vector, faiss_index, k):
    """
    Search for k most similar images to a query vector.

    Args:
    - query_vector: Feature vector of the query image.
    - faiss_index: The FAISS index to use for searching.
    - k: Number of similar images to find.

    Returns:
    - Indices of similar images in the FAISS index.
    """
    query_vector = np.array(query_vector, dtype='float32').reshape(1, -1)
    _, indices = faiss_index.search(query_vector, k)
    return indices[0]


def load_vectors_and_keys(pt_path):
    """
    Load feature vectors and their keys from a .pt file.

    Args:
    - pt_path: Path to the .pt file containing vectors and keys.

    Returns:
    - A tuple containing the vectors and their corresponding keys.
    """
    output_dict = torch.load(pt_path, map_location=torch.device('cpu'))
    keys = list(output_dict.keys())
    vectors = np.stack([output_dict[key] for key in keys])  
    return vectors, keys



def save_faiss_index(index, index_path):
    """
    Save the FAISS index to disk.

    Args:
    - index: The FAISS index to save.
    - index_path: Path to the file where to save the index.
    """
    faiss.write_index(index, index_path)
    

def create_and_save_mapping(keys, mapping_path):
    """
    Create and save a mapping between FAISS indices and keys.

    Args:
    - keys: The keys corresponding to the feature vectors.
    - mapping_path: Path to the JSON file where to save the mapping.
    """
    index_to_key_mapping = {str(i): key for i, key in enumerate(keys)}
    with open(mapping_path, 'w') as f:
        json.dump(index_to_key_mapping, f)
        

def load_index_to_path_mapping(mapping_json_path):
    """
    Load the mapping between FAISS indices and image paths.

    Args:
    - mapping_json_path: Path to the JSON file containing the mapping.

    Returns:
    - The loaded mapping.
    """
    with open(mapping_json_path, 'r') as f:
        return json.load(f)
    

def show_similar_pics_faiss(request, input_pic, model, nbr_show, faiss_index_path):
    # Record start time to calculate processing time later
    start_time = time.time()

    # Load the FAISS index from the given path
    faiss_index = load_faiss_index(faiss_index_path)

    # Load the mapping between FAISS indices and image paths
    index_to_path_mapping = load_index_to_path_mapping(settings.FAISS_MAPPING_PATH_AVGPOOL)

    # Get the feature vector for the input image using the specified model
    vector = get_vector(input_pic, model)

    # Apply PCA transformation if the model is one of these RESNET18 layers
    if model == "RESNET18_LAYER4":
        pca = load(settings.PCA_PATH_LAYER4)
        print(pca.n_features_in_)
        vector = vector.numpy().flatten()
        vector = pca.transform(vector.reshape(1, -1))
        vector = vector.astype('float32')

    if model == "RESNET18_LAYER3":
        #pca = load("/home/yassfkh/Desktop/L3H1/2023-l3h1/branches/testYassine/WebApp/ProjetCBIR/cnn/output_dict_wikiart_pca_layer3.joblib")
        pca = load(settings.PCA_PATH_LAYER3)
        print(pca.n_features_in_)
        vector = vector.numpy().flatten()
        vector = pca.transform(vector.reshape(1, -1))
        vector = vector.astype('float32')

    if model == "RESNET18_LAYER2":
        #pca = load("/home/yassfkh/Desktop/L3H1/2023-l3h1/branches/testYassine/WebApp/ProjetCBIR/cnn/output_dict_wikiart_pca_layer2.joblib")
        pca = load(settings.PCA_PATH_LAYER2)
        print(pca.n_features_in_)
        vector = vector.numpy().flatten()
        vector = pca.transform(vector.reshape(1, -1))
        vector = vector.astype('float32')

    # Search for similar images in the FAISS index
    similar_indices = search_similar_images_faiss(vector, faiss_index, nbr_show)

    # Build the absolute URL for each similar image and store it in a list
    similarities_show = [request.build_absolute_uri(settings.MEDIA_URL + index_to_path_mapping[str(idx)]) for idx in similar_indices]

    # Calculate the processing time
    similarity_time = time.time() - start_time

    # Return the list of similar image URIs and the processing time
    return similarities_show, similarity_time





#  ! UNCOMMENT THESE LINES ONLY IF YOU WANT TO RECREATE INDEXES AND MAPPING FILES !

""" feature_vectors, keys = load_vectors_and_keys(VECTORS_PT_PATH)
print("Feature vectors shape:", feature_vectors.shape)
feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)
print("Feature vectors shape:", feature_vectors.shape)
faiss_index = create_faiss_index(feature_vectors)
save_faiss_index(faiss_index, FAISS_INDEX_PATH)
create_and_save_mapping(keys, MAPPING_JSON_PATH) """