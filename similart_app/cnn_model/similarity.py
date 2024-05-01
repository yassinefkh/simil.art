"""
This file contains functions for image similarity retrieval and displaying similar images.
"""

import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import time
from joblib import load
from django.conf import settings
from cnn_model.features_extraction import get_vector


# Path to the folder containing the images
db_path = '/media/'


def similarity(input_pic, model):
    """
    Calculate cosine similarity between the input picture and images in the dataset.

    Args:
    - input_pic: Path to the input image.
    - model: Name of the model to use for feature extraction.

    Returns:
    - List of indices of similar images.
    """
    pic_vector = get_vector(input_pic, model)  # feature vector of the input image
    similarities = []  # list of similarities

    # Load PCA model based on the specified layer
    match model:
        case "RESNET18_LAYER4":
            pca = load(settings.PCA_PATH_LAYER4) 
            dict_layer = load(settings.FEATURES_VECTORS_PATH_LAYER4) 
        case "RESNET18_LAYER3":
            pca = load(settings.PCA_PATH_LAYER3)
            dict_layer = load(settings.FEATURES_VECTORS_PATH_LAYER3) 
        case "RESNET18_LAYER2":
            pca = load(settings.PCA_PATH_LAYER2)
            dict_layer = load(settings.FEATURES_VECTORS_PATH_LAYER2) 
        case _:
            dict_layer = load(settings.FEATURES_VECTORS_PATH_AVGPOOL) 

    pic_vector_pca = pca.transform(pic_vector.numpy().flatten().reshape(1, -1))

    # Calculate cosine similarity for each image in the dataset
    for file, output_vector in dict_layer.items():
        cos_sim = cosine_similarity(pic_vector_pca, output_vector.reshape(1, -1))
        similarities.append(cos_sim[0][0])

    # Sort similarities
    sorted_indices = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)

    return sorted_indices


def show_similar_pics(request, input_pic, model, nbr_show):
    """
    Display the most similar images to a query image.

    Args:
    - request: Django HTTP request object.
    - input_pic: Path to the input image.
    - model: Name of the model to use for feature extraction.
    - nbr_show: Number of similar images to display.

    Returns:
    - List of URLs of similar images.
    - Time taken for similarity calculation.
    """
    # Select dictionary based on the specified model
    match model:
        case "RESNET18_LAYER4":
            dict = load(settings.FEATURES_VECTORS_PATH_LAYER4) 
        case _:
            dict = load(settings.FEATURES_VECTORS_PATH_AVGPOOL) 

    similarities_show = []
    start_time = time.time()  # Start time
    sorted_indices = similarity(input_pic, model)  # Call similarity function
    end_time = time.time()  # End time
    similarity_time_1 = end_time - start_time  # Calculate time for similarity function
    similarity_time = round(similarity_time_1, 2)

    # Display the most similar images
    for i in range(nbr_show):
        image_path = list(dict.keys())[sorted_indices[i]]  # Get image path
        image_path = db_path + image_path  # Add path to image folder
        image_path = image_path.replace('\\', '/')  # Replace backslashes with slashes

        # Change path to be accessible by the server
        image_url = image_path.replace(db_path, settings.MEDIA_URL)
        image_url = request.build_absolute_uri(image_url)

        similarities_show.append(image_url)  # Add image URLs to the list

    return similarities_show, similarity_time
