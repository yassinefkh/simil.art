import pytest
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.messages import get_messages 
from app.views import index, home, start, search, equipe, help_histogram, image_retrieval, upload_image, upload_image_faiss, show_feature_maps, refine_results, download_similar_images
from django.contrib.auth.models import User
from django.test import RequestFactory, Client
from django.contrib.sessions.middleware import SessionMiddleware
from django.contrib.messages.middleware import MessageMiddleware
from django.utils.deprecation import MiddlewareMixin

class DummyGetResponseMiddleware(MiddlewareMixin):
    """
    A dummy middleware class to simulate the behavior of get_response.
    """
    def __init__(self, get_response=None):
        self.get_response = get_response

@pytest.mark.django_db
def test_home_view(client):
    """
    Test the home view.
    Verifies if the view returns an HTTP response with a status code of 200 (OK)
    and if the template used for the response is 'home.html'.
    """
    url = reverse('home')
    response = client.get(url)
    assert response.status_code == 200
    assertTemplateUsed(response, 'home.html')

@pytest.mark.django_db
def test_start_view(client):
    """
    Test the start view.
    Verifies if the view returns an HTTP response with a status code of 200 (OK)
    and if the template used for the response is 'start.html'.
    """
    url = reverse('start')
    response = client.get(url)
    assert response.status_code == 200
    assertTemplateUsed(response, 'start.html')

def assertTemplateUsed(response, template_name):
    """
    Helper function to mimic Django's TestCase `assertTemplateUsed` method.
    """
    assert template_name in [t.name for t in response.templates]

@pytest.mark.django_db
def test_search_view(client):
    """
    Test the search view.
    Verifies if the view returns an HTTP response with a status code of 200 (OK)
    and if the response contains 'Search'.
    """
    url = reverse('search')
    response = client.get(url)
    assert response.status_code == 200
    assert 'Search' in response.content.decode()  

@pytest.mark.django_db
def test_upload_image_faiss(client):
    """
    Test the upload_image_faiss view.
    Verifies if the image has been uploaded successfully and if the view returns a status code of 200 (OK).
    """
    url = reverse('upload_image_faiss')
    with open('/home/yassfkh/Desktop/L3H1/2023-l3h1/branches/testYassine/WebApp/ProjetCBIR/app/media/wikiartresized/Action_painting/franz-kline_accent-grave-1955.jpg', 'rb') as img:
        response = client.post(url, {'image': SimpleUploadedFile(img.name, img.read(), content_type='image/jpeg')})
        assert response.status_code == 200
        assert 'The image has been uploaded successfully.' in response.content.decode()

@pytest.mark.django_db       
def test_upload_image(client):
    """
    Test the upload_image view.
    Verifies if the image has been uploaded successfully and if the view returns a status code of 200 (OK).
    """
    url = reverse('upload_image')
    with open('/home/yassfkh/Desktop/L3H1/2023-l3h1/branches/testYassine/WebApp/ProjetCBIR/app/media/wikiartresized/Action_painting/franz-kline_accent-grave-1955.jpg', 'rb') as img:
        response = client.post(url, {'image': SimpleUploadedFile(img.name, img.read(), content_type='image/jpeg')})
        assert response.status_code == 200
        assert 'The image has been uploaded successfully.' in response.content.decode()

@pytest.mark.django_db
def test_upload_image_invalid_format(client):
    """
    Test the upload_image_faiss view with an invalid image format.
    Verifies if the view correctly handles an invalid image format.
    """
    url = reverse('upload_image_faiss')
    data = {'image': SimpleUploadedFile('/home/yassfkh/Desktop/L3H1/2023-l3h1/branches/testYassine/WebApp/ProjetCBIR/app/testFront/incorrectFile.txt', b'This is not an image', content_type='text/plain')}
    response = client.post(url, data)
    messages = [msg.message for msg in get_messages(response.wsgi_request)]
    assert any("Invalid image format." in message for message in messages), "Expected error message not found."

@pytest.mark.django_db
def test_show_feature_maps_with_uploaded_image():
    """
    Test the show_feature_maps view with an uploaded image.
    Verifies if the view correctly redirects to the search page when an image is uploaded.
    """
    # Create a user to simulate a session
    user = User.objects.create_user(username='testuser', password='12345')

    # Set up the session with an uploaded image
    request = RequestFactory().get(reverse('feature_maps'))
    request.user = user

    # Initialize the SessionMiddleware with a dummy get_response function
    middleware = SessionMiddleware(lambda x: None)
    middleware.process_request(request)
    request.session['uploaded_image'] = '/home/yassfkh/Desktop/L3H1/2023-l3h1/branches/testYassine/WebApp/ProjetCBIR/app/media/wikiartresized/Action_painting/franz-kline_accent-grave-1955.jpg'
    request.session.save()

    # Use the client to call the show_feature_maps view
    client = Client()
    response = client.get(reverse('feature_maps'))

    # Verify that the response is a redirection
    assert response.status_code == 302
    # Verify that the redirection is correct
    assert response.url == reverse('search')

@pytest.mark.django_db
def test_show_feature_maps_without_uploaded_image(client):
    """
    Test the show_feature_maps view without an uploaded image.
    Verifies if the view correctly redirects to the search page and adds an error message to the session.
    """
    # Simulate no image uploaded in the session
    if 'uploaded_image' in client.session:
        del client.session['uploaded_image']
    client.session.save()

    # Make a GET request to the show_feature_maps view
    url = reverse('feature_maps')
    response = client.get(url)

    # Assert that the response status code is 302 (redirection)
    assert response.status_code == 302

    # Assert that the error message is added to the session
    messages = [msg.message for msg in get_messages(response.wsgi_request)]
    assert 'Aucune image n\'a été téléversée.' in messages
