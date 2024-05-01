"""
This file contains functions for plotting t-SNE visualizations and visualizing feature maps.
"""

import os
import numpy as np
from PIL import Image
from django.conf import settings
from sklearn.manifold import TSNE
from joblib import load
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import torchvision.transforms as transforms
from cnn_model.model import model_resnet18_avgpool, model_resnet18_layer3, model_resnet18_layer2, model_resnet18_layer4
from indexing.faiss_indexing import load_faiss_index, load_index_to_path_mapping, search_similar_images_faiss
from cnn_model.features_extraction import get_vector

# Path to the CSV file containing image classes
csv_file_path = os.path.join(settings.BASE_DIR, 'wikiartresized', 'classes.csv')

# Path to the folder containing the images
db_path = '/media/'


def plot_tsne(input_image_path, model, nbr_show):
    """
    Plot t-SNE visualization of similar images to the input image.

    Args:
    - input_image_path: Path to the input image.
    - model: Name of the model to use for feature extraction.
    - nbr_show: Number of similar images to plot.

    Returns:
    - Plotly figure for t-SNE visualization.
    """
    # Determine the appropriate paths based on the selected model
    match model:
        case "RESNET18_LAYER2":
            faiss_index_path = settings.FAISS_INDEX_PATH_LAYER2
            mapping_json_path = settings.FAISS_MAPPING_PATH_LAYER2
            pca_model_path = settings.PCA_PATH_LAYER2
        case "RESNET18_LAYER3":
            faiss_index_path = settings.FAISS_INDEX_PATH_LAYER3
            mapping_json_path = settings.FAISS_MAPPING_PATH_LAYER3
            pca_model_path = settings.PCA_PATH_LAYER3
        case "RESNET18_LAYER4":
            faiss_index_path = settings.FAISS_INDEX_PATH_LAYER4
            mapping_json_path = settings.FAISS_MAPPING_PATH_LAYER4
            pca_model_path = settings.PCA_PATH_LAYER4
        case _:
            faiss_index_path = settings.FAISS_INDEX_PATH_AVGPOOL
            mapping_json_path = settings.FAISS_MAPPING_PATH_AVGPOOL
            pca_model_path = None

    # Load the FAIS index and the index-to-path mapping
    faiss_index = load_faiss_index(faiss_index_path)
    index_to_path_mapping = load_index_to_path_mapping(mapping_json_path)

    # Get the input vector for the input image and flatten it
    input_vector = get_vector(input_image_path, model).numpy().flatten().reshape(1, -1)

    # If a PCA model path is provided, apply PCA to the input vector
    if pca_model_path:
        pca = load(pca_model_path)
        input_vector = pca.transform(input_vector)

    # Find the indices of the most similar images using the FAIS index
    similar_indices = search_similar_images_faiss(input_vector, faiss_index, nbr_show + 1)

    # Create lists to store the vectors and labels for the most similar images
    vectors = [input_vector]
    labels = [input_image_path]

    # Loop through the indices of the most similar images (excluding the input image)
    for idx in similar_indices[1:]:
        # Get the path of the similar image
        similar_image_path = index_to_path_mapping[str(idx)]
        # Get the vector for the similar image and flatten it
        similar_vector = get_vector("core_interface/media/" + similar_image_path, model).numpy().flatten().reshape(1, -1)
        # If a PCA model path is provided, apply PCA to the similar vector
        if pca_model_path:
            similar_vector = pca.transform(similar_vector)
        # Add the similar vector and path to the vectors and labels lists
        vectors.append(similar_vector)
        labels.append(similar_image_path)

    # Convert the vectors list to a NumPy array
    vectors = np.vstack(vectors)

    # Apply t-SNE to the vectors to reduce their dimensionality to 2
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors)

    # Create a Plotly figure
    fig = go.Figure()

    # Define a list of colors for each category
    categories = csv_file_path['genre']
    cmap = plt.cm.get_cmap('rainbow', len(categories))
    color_dict = {category: mcolors.rgb2hex(cmap(i)) for i, category in enumerate(categories)}

    # Create a dictionary to store the plots for each category
    category_plots = {category: [] for category in categories}

    # Loop through the 2D vectors (excluding the input image)
    for i in range(1, len(vectors_2d)):
        # Get the category of the image
        category = labels[i].split('/')[1]
        # Get the color for the category
        color = color_dict.get(category, 'black')
        # Get the image URL
        image_url = labels[i].replace('app/media/', '/media/')

        # Create the hover template
        hovertemplate = f'<img src="{image_url}" width="32" height="32"><br>%{{extra}}'

        # If the category is not already in the category_plots dictionary, add it
        if category not in category_plots:
            category_plots[category] = []

        # Add a scatter plot for the image to the category_plots dictionary
        category_plots[category].append(go.Scatter(x=[vectors_2d[i, 0]], y=[vectors_2d[i, 1]], mode='markers', marker=dict(color=color), name=category, hovertext=category, hovertemplate=hovertemplate, customdata=[category]))

    # Add the scatter plots for each category to the figure
    for plots in category_plots.values():
        for plot in plots:
            fig.add_trace(plot)

    fig.add_trace(go.Scatter(x=[vectors_2d[0, 0]], y=[vectors_2d[0, 1]], mode='markers', marker=dict(color='purple', size=20, symbol='star')))

    fig.update_layout(title="t-SNE Visualization of Similar Images",
                      xaxis_title="",
                      yaxis_title="")

    return fig


def visualize_feature_maps(image_url, model_name, layer_name):
    """
    Visualize feature maps of a specific layer in the model for an input image.

    Args:
    - image_url: URL of the input image.
    - model_name: Name of the model to use for feature extraction.
    - layer_name: Name of the layer to visualize feature maps for.

    Returns:
    - HTML representation of the feature maps.
    """
    img = Image.open(image_url).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)

    if model_name == "RESNET18_AVGPOOL":
       model, _ = model_resnet18_avgpool()
    elif model_name == "RESNET18_LAYER3":
        model, _ = model_resnet18_layer3()
    elif model_name == "RESNET18_LAYER2":
        model, _ = model_resnet18_layer2()
    elif model_name == "RESNET18_LAYER4":
        model, _ = model_resnet18_layer4()
    else:
        raise ValueError("Invalid model name")

    model.eval()

    # pass the image through the model and get the feature maps of the specified layer
    feature_maps = None
    for name, layer in model.named_children():
        img_tensor = layer(img_tensor) # apply the layer to the tensor
        if name == layer_name:
            feature_maps = img_tensor.squeeze().detach().numpy()  # extract the feature maps
            break # exit the loop once the desired layer is reached
    
    if feature_maps is None:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # normalize the feature maps
    feature_maps -= feature_maps.min()
    feature_maps /= feature_maps.max()

    # Create the Plotly figure
    n_maps = feature_maps.shape[0]
    cols = 6
    rows = n_maps // cols + (1 if n_maps % cols else 0) 

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'Feature map {i+1}' for i in range(n_maps)])

    for i in range(n_maps):
        row = i // cols + 1
        col = i % cols + 1
        fig.append_trace(go.Heatmap(z=feature_maps[i], colorscale='Viridis'), row=row, col=col)
        
    fig.update_traces(showscale=False) # to remove the colorbar
    fig.update_layout(height=200*rows, width=200*cols, title_text=f'Feature maps for layer {layer_name}')
    feature_map_html = fig.to_html(full_html=False)

    return feature_map_html
