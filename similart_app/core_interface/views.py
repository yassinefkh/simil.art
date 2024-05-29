from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import default_storage
from django.contrib import messages
from django.shortcuts import redirect

import os
from io import BytesIO
from urllib.parse import urlparse
import zipfile

import plotly
import plotly.offline as opy

from cnn_model.visualization import plot_tsne, visualize_feature_maps
from cnn_model.similarity import similarity, show_similar_pics
from image_analysis.color_analysis import find_similar_images
from image_analysis.style_analysis import find_similar_style_images
from indexing.faiss_indexing import load_faiss_index, show_similar_pics_faiss

from core_interface.utils import url_to_local_path

"""

Views for handling HTTP requests and rendering HTML templates.


"""


def index(request):
    """
    View for the homepage.
    This view returns an HTTP response with the message "Hello, world."
    Parameters:
    - request (HttpRequest): The HttpRequest object for the HTTP request.
    Returns:
    - HttpResponse: An HTTP response with the message "Hello, world."
    """
    return HttpResponse("Hello, world.")

def home(request):
    """
    View for the 'home' page.
    This view returns the rendering of the 'home.html' template.
    Parameters:
    - request (HttpRequest): The HttpRequest object for the HTTP request.
    Returns:
    - HttpResponse: The rendering of the 'home.html' template.
    """
    return render(request, 'home.html')

def start(request):
    """
    View for the 'start' page.
    This view returns the rendering of the 'start.html' template.
    Parameters:
    - request (HttpRequest): The HttpRequest object for the HTTP request.
    Returns:
    - HttpResponse: The rendering of the 'start.html' template.
    """
    return render(request, 'start.html')

def search(request):
    """
    View for the 'search' page.
    This view returns the rendering of the 'search.html' template.
    Parameters:
    - request (HttpRequest): The HttpRequest object for the HTTP request.
    Returns:
    - HttpResponse: The rendering of the 'search.html' template.
    """
    return render(request, 'search.html')

def equipe(request):
    """
    View for the 'equipe' page.
    This view returns the rendering of the 'equipe.html' template.
    Parameters:
    - request (HttpRequest): The HttpRequest object for the HTTP request.
    Returns:
    - HttpResponse: The rendering of the 'equipe.html' template.
    """
    return render(request, 'equipe.html')

def help_histogram(request):
    """
    View for the 'help-histogram' page.
    This view returns the rendering of the 'help_histogram.html' template.
    Parameters:
    - request (HttpRequest): The HttpRequest object for the HTTP request.
    Returns:
    - HttpResponse: The rendering of the 'help_histogram.html' template.
    """
    return render(request, 'help_histogram.html')

def image_retrieval(request):
    """
    View for the 'help-histogram' page.
    This view returns the rendering of the 'help_histogram.html' template.
    Parameters:
    - request (HttpRequest): The HttpRequest object for the HTTP request.
    Returns:
    - HttpResponse: The rendering of the 'help_histogram.html' template.
    """
    return render(request, 'help_image-retrieval.html')

    
def upload_image(request):
    """
    Handle image upload and similarity search.

    Parameters:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: Response with search results.
    """
    # Check if the request method is POST and an image is uploaded
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image
        uploaded_image = request.FILES['image']
        # Get the selected layer for feature extraction
        selected_layer = request.POST.get('layer', 'avgpool')

        # Store the uploaded image in a temporary folder within the media directory
        filename = default_storage.save('temp/' + uploaded_image.name, uploaded_image)
        messages.success(request, 'The image has been uploaded successfully.')

        # Perform image search based on the selected layer
        match selected_layer:
            case "avgpool":
                # Perform image search using average pooling features
                similar_images_paths, similarity_time = show_similar_pics(request, uploaded_image, "RESNET18_AVGPOOL", 300)
            case "layer4":
                # Perform image search using layer 4 features
                similar_images_paths, similarity_time = show_similar_pics(request, uploaded_image, "RESNET18_LAYER4", 300)
        
        # Print the paths of similar images (for debugging purposes)
        for path in similar_images_paths:
            print(path)

        # Initialize variables for optional visualizations
        tsne_plot_html = ""
        feature_map_html = ""

        # Store relevant information in the session
        request.session['similar_images_paths'] = similar_images_paths
        request.session['uploaded_image_name'] = uploaded_image.name
        request.session['selected_layer'] = selected_layer

        # Check if the user requested t-SNE plot visualization
        if 'show_tsne' in request.POST:
            tsne_plot = plot_tsne(uploaded_image, "RESNET18_AVGPOOL", 300)
            tsne_plot_html = plotly.offline.plot(tsne_plot, output_type='div')   
            
        # Check if the user requested feature map visualization
        if 'show_featuremap' in request.POST:
            feature_map_fig = visualize_feature_maps(uploaded_image, "RESNET18_AVGPOOL", "conv1")
            feature_map_html = opy.plot(feature_map_fig, auto_open=False, output_type='div')
            
        # Prepare context for rendering the search results page
        context = {
            'similar_images_paths': similar_images_paths,
            'uploaded_image': settings.MEDIA_URL + 'temp/' + uploaded_image.name,
            'feature_map_plot': feature_map_html,
            'tsne_plot': tsne_plot_html,
            'similarity_time': similarity_time,
        }

        # Render the search results page with the context
        return render(request, 'search.html', context)
    else:
        # If the request method is not POST or no image is uploaded, show an error message and redirect to the search page
        messages.error(request, 'No images have been uploaded.')
        return redirect('search.html')



# UPLOAD IMAGE MAIS AVEC FAISS
def upload_image_faiss(request):
    """
    Handle image upload and search using FAISS index.

    Parameters:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: Response with search results.
    """
    # Check if the request method is POST
    if request.method == 'POST':
        # Check if an image has been uploaded
        if not request.FILES.get('image'):
            # If no image is uploaded, show an error message and redirect to the search page
            messages.error(request, "No images have been uploaded.")
            return redirect('search')
        else:
            # Get the uploaded image
            uploaded_image = request.FILES['image']

            # Check if the uploaded file is an image with a valid format
            allowed_extensions = ['jpg', 'jpeg', 'png', 'gif']
            file_extension = uploaded_image.name.split('.')[-1].lower()

            if file_extension not in allowed_extensions:
                # If the image format is invalid, show an error message and redirect to the search page
                messages.error(request, "Invalid image format. Please upload an image in the following formats: " + ', '.join(allowed_extensions))
                return redirect('search')

            # Get the selected layer for feature extraction
            selected_layer = request.POST.get('layer', 'avgpool')

            # Store the uploaded image in a temporary folder within the media directory
            filename = default_storage.save('temp/' + uploaded_image.name, uploaded_image)
            messages.success(request, 'The image has been uploaded successfully.')

            # Determine the FAISS index path and the corresponding model name based on the selected layer
            faiss_index_path = None
            model_name = None
            match selected_layer:
                case "avgpool":
                    faiss_index_path = settings.FAISS_INDEX_PATH_AVGPOOL
                    model_name = "RESNET18_AVGPOOL"
                case "layer4":
                    faiss_index_path = settings.FAISS_INDEX_PATH_LAYER4
                    model_name = "RESNET18_LAYER4"
                case "layer3":
                    faiss_index_path = settings.FAISS_INDEX_PATH_LAYER3
                    model_name = "RESNET18_LAYER3"
                case "layer2":
                    faiss_index_path = settings.FAISS_INDEX_PATH_LAYER2
                    model_name = "RESNET18_LAYER2"

            # Check if a valid FAISS index path is found
            if faiss_index_path:
                # Load the FAISS index
                faiss_index = load_faiss_index(faiss_index_path)

                # Get the dimensionality of the FAISS index
                index_dimensionality = faiss_index.d
                print("Dimensionality of FAISS index:", index_dimensionality)

                # Perform image search using the FAISS index
                similar_images_paths, similarity_time = show_similar_pics_faiss(request, uploaded_image, model_name, 100, faiss_index_path)

                # Store relevant information in the session
                request.session['similar_images_paths'] = similar_images_paths
                request.session['uploaded_image_name'] = uploaded_image.name
                request.session['uploaded_image'] = default_storage.path(filename)
                request.session['selected_layer'] = selected_layer

                # Initialize variables for optional visualizations
                tsne_plot_html = ""
                feature_map_html = ""

                # Check if the user requested t-SNE plot visualization
                if 'show_tsne' in request.POST:
                    tsne_plot = plot_tsne(uploaded_image, model_name, 50)
                    tsne_plot_html = plotly.offline.plot(tsne_plot, output_type='div')

                # Check if the user requested feature map visualization
                if 'show_featuremap' in request.POST:
                    feature_map_fig = visualize_feature_maps(uploaded_image, model_name, "conv1")
                    feature_map_html = opy.plot(feature_map_fig, auto_open=False, output_type='div')

                # Prepare context for rendering the search results page
                context = {
                    'similar_images_paths': similar_images_paths,
                    'uploaded_image': settings.MEDIA_URL + 'temp/' + uploaded_image.name,
                    'feature_map_plot': feature_map_html,
                    'tsne_plot': tsne_plot_html,
                    'similarity_time': similarity_time,
                }

                # Render the search results page with the context
                return render(request, 'search.html', context)
            else:
                # If no appropriate FAISS index path is found, show an error message and redirect to the search page
                messages.error(request, 'No appropriate FAISS index found.')
                return redirect('search.html')
    else:
        # If the request method is not POST, show an error message and redirect to the search page
        messages.error(request, 'No images have been uploaded.')
        return redirect('search.html')

    
    
def show_feature_maps(request):
    """
    Display feature maps for the uploaded image.

    Parameters:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: Response with the feature map visualization.
    """
    # Check if an image has been uploaded
    if 'uploaded_image' not in request.session:
        # If no image is uploaded, show an error message and redirect to the search page
        messages.error(request, 'Aucune image n\'a été téléversée.')
        return redirect('search')

    # Get the path of the uploaded image from the session
    uploaded_image_path = request.session['uploaded_image']
    # Get the selected layer from the session, defaulting to 'avgpool'
    selected_layer = request.session.get('selected_layer', 'avgpool')

    # Determine the model name based on the selected layer
    model_name = None
    if selected_layer == "avgpool":
        model_name = "RESNET18_AVGPOOL"
    elif selected_layer == "layer4":
        model_name = "RESNET18_LAYER4"
    elif selected_layer == "layer3":
        model_name = "RESNET18_LAYER3"
    elif selected_layer == "layer2":
        model_name = "RESNET18_LAYER2"

    # If a model name is found
    if model_name:
        # Visualize the feature maps for the uploaded image at the 'conv1' layer
        feature_map_html = visualize_feature_maps(uploaded_image_path, model_name, "conv1")

        # Prepare the context with the feature map visualization
        context = {
            'feature_map_plot': feature_map_html,
        }

        # Render the feature map page with the context
        return render(request, 'feature_map_page.html', context)
    else:
        # If no appropriate model name is found, show an error message and redirect to the search page
        messages.error(request, 'Aucun modèle approprié trouvé.')
        return redirect('search')





def refine_results(request):
    """
    View function to refine search results based on color similarity.
    
    Parameters:
        request (HttpRequest): The HTTP request object.
        
    Returns:
        HttpResponse: Rendered HTML response with refined search results.
    """
    if request.method == 'POST':
        try:
            # Check if necessary session data is available
            if 'similar_images_paths' in request.session and 'uploaded_image_name' in request.session and 'selected_layer' in request.session:
                # Retrieve session data
                similar_images_paths = request.session.get('similar_images_paths')
                uploaded_image_name = request.session.get('uploaded_image_name')
                selected_layer = request.session['selected_layer']

                # Get the path of the uploaded image
                uploaded_image_path = os.path.join(settings.MEDIA_ROOT, 'temp/', uploaded_image_name)

                # Get the threshold value from the POST data
                threshold = float(request.POST.get('threshold'))

                # Find similar images based on the uploaded image
                results = find_similar_images(uploaded_image_path, similar_images_paths, threshold=threshold)

                # Extract the color information from the results
                similar_image_color = [result[0] for result in results]

                # Prepare context data to pass to the template
                context = {
                    'similar_image_color': similar_image_color,
                    'total_images_count': len(similar_images_paths),
                    'filtered_images_count': len(similar_image_color),
                }
                # Store similar image color data in session for future use
                request.session['similar_image_color'] = similar_image_color

                # Render the refine_results.html template with the context data
                return render(request, 'refine_results.html', context)
            else:
                # If necessary session data is not available, display an error message
                messages.error(request, "The necessary information is not available in the session.")
                return render(request, 'refine_results.html')
        except Exception as e:
            # If an exception occurs during processing, display an error message
            messages.error(request, "An error occurred while processing your request. Please try again later.")
            return render(request, 'refine_results.html')
    else:
        # If the request method is not POST, return a bad request response
        return HttpResponseBadRequest("This view only supports POST requests.")
    

def style_analysis(request):

    """

    View function to refine search results based on the syle of art. 

    

    Parameters:

        request (HttpRequest): The HTTP request object.

        

    Returns:

        HttpResponse: Rendered HTML response with refined search results.

    """

    if request.method == 'POST':

        try:

            # Check if necessary session data is available

            if 'similar_images_paths' in request.session and 'uploaded_image_name' in request.session and 'selected_layer' in request.session:

                # Retrieve session data

                similar_images_paths = request.session.get('similar_images_paths')

                uploaded_image_name = request.session.get('uploaded_image_name')

                # Get the path of the uploaded image

                uploaded_image_path = os.path.join(settings.MEDIA_ROOT, 'temp/', uploaded_image_name)

                # Find similar images based on the uploaded image

                results, upload_style = find_similar_style_images(uploaded_image_path, similar_images_paths)

                # Extract the color information from the results

                similar_image_style = results

                # Prepare context data to pass to the template

                context = {

                    'upload_style': upload_style,

                    'similar_image_style': similar_image_style,

                    'total_images_count': len(similar_images_paths),

                    'filtered_images_count': len(similar_image_style),

                }

               

                # Render the refine_results.html template with the context data

                return render(request, 'refine_by_style.html', context)

            else:

                # If necessary session data is not available, display an error message

                messages.error(request, "The necessary information is not available in the session.")

                return render(request, 'refine_by_style.html')

        except Exception as e:

            # If an exception occurs during processing, display an error message

            messages.error(request, "An error occurred while processing your request. Please try again later.")

            return render(request, 'refine_by_style.html')

    else:

        # If the request method is not POST, return a bad request response

        return HttpResponseBadRequest("This view only supports POST requests.")



def download_similar_images(request):
    """
    Download similar images as a zip file.

    Parameters:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: Response with the zip file containing similar images.
    """
    # Get the paths of similar images from the session
    similar_images_paths = request.session.get('similar_images_paths', [])
    
    # Create an in-memory zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # Iterate over the image URLs and add them to the zip file
        for image_url in similar_images_paths:
            # Convert the URL to a local file path
            image_path = url_to_local_path(image_url, settings.MEDIA_ROOT)
            # Add the image to the zip file
            zip_file.write(image_path, os.path.basename(image_path))
    
    # Set the file pointer to the beginning of the buffer
    zip_buffer.seek(0)
    
    # Create an HTTP response with the zip file as the content
    response = HttpResponse(zip_buffer, content_type='application/zip')
    # Set the content disposition header for downloading as a file
    response['Content-Disposition'] = 'attachment; filename=similar_images.zip'
    
    return response

def terms_of_use(request):
    """
    show the terms of use page
    Parameters:
        request (HttpRequest): The HTTP request object.
    Returns:
        HttpResponse: The rendering of the 'terms_of_use.html' template.
    """
    return render(request, 'terms_of_use.html')