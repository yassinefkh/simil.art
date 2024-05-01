import os
from urllib.parse import urlparse


def url_to_local_path(url, media_root):
    """
    Convert a URL to a local file path.
    
    Parameters:
        url (str): The URL of the file.
        media_root (str): The root directory of media files in the Django project.
        
    Returns:
        str: The local file path corresponding to the URL.
    """
    # Parse the URL to extract the path
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Remove the MEDIA_URL prefix from the path
    if path.startswith(settings.MEDIA_URL):
        path = path[len(settings.MEDIA_URL):]
    
    # Extract the artist name and file name from the path
    artist_name = path.split('/')[1]
    file_name = path.split('/')[-1]
    
    # Construct the local file path using the artist name and file name
    local_path = os.path.join(media_root, 'wikiartresized', artist_name, file_name)
    
    return local_path