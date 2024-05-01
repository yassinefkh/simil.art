import pytest
import os
import cv2
import numpy as np
from django.conf import settings
from image_analysis.color_analysis import calculate_color_histogram, calculate_correlation, find_similar_images

# Path to the test images folder
TEST_IMAGES_PATH = os.path.join(settings.BASE_DIR, 'cnn_model','tests')

def test_calculate_color_histogram():
    """
    Test the calculate_color_histogram function.
    """
    # Load a test image
    img_path = f"{TEST_IMAGES_PATH}/test_image.jpg"
    img = cv2.imread(img_path)

    # Calculate color histogram
    hist = calculate_color_histogram(img)

    # Check that the output is a NumPy array with the expected shape
    assert isinstance(hist, np.ndarray)
    assert hist.shape == (180,)

def test_calculate_correlation():
    """
    Test the calculate_correlation function.
    """
    # Create two sample color histograms
    hist1 = np.random.rand(180)
    hist2 = np.random.rand(180)

    # Calculate correlation
    correlation = calculate_correlation(hist1, hist2)

    # Check that the output is a float between 0 and 1
    assert isinstance(correlation, float)
    assert 0 <= correlation <= 1


def test_find_similar_images():
    """
    Test the find_similar_images function.
    """
    # Load a test image
    img_path = f"{TEST_IMAGES_PATH}/test_image.jpg"

    # Find similar images
    similar_images = find_similar_images(img_path, threshold=0.8)

    # Check that the output is a list of tuples containing similar image URLs and images
    assert all(isinstance(url, str) and isinstance(img, np.ndarray) for url, img in similar_images)
