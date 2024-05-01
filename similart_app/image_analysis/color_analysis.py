import cv2
import os
import matplotlib.pyplot as plt
import urllib.parse
from django.conf import settings
import requests
import numpy as np

from core_interface.utils import url_to_local_path

"""
File to find similar images based on color histogram.
"""

def calculate_color_histogram(image):
    """
    Calculate and return the color histogram of an image in HSV color space.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the color histogram for the hue channel (H)
    hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()

    #print("Color histogram calculated")
    return hist

def calculate_correlation(hist1, hist2):
    """
    Calculate and return the correlation between two color histograms.
    """
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    #print("Correlation calculated")
    return correlation

def find_similar_images_test(query_image, test_images_folder, threshold=0.8):
    """
    Find and return similar images to the query image based on color histogram comparison.
    """
    query_image = cv2.imread(query_image)
    if query_image is None:
        print(f"ERROR: Unable to read image '{query_image}'.")
        return

    # Calculate the color histogram of the query image
    query_hist = calculate_color_histogram(query_image)

    similar_images = []

    # Iterate through all images in the test folder
    for test_image_name in os.listdir(test_images_folder):
        test_image_path = os.path.join(test_images_folder, test_image_name)
        if test_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_image = cv2.imread(test_image_path)
            if test_image is None:
                print(f"ERROR: Unable to read image '{test_image_path}'.")
                continue

            # Calculate the color histogram of the test image
            test_hist = calculate_color_histogram(test_image)

            # Calculate the correlation between the query image and test image color histograms
            correlation = calculate_correlation(query_hist, test_hist)

            # Check if the test image is similar to the query image based on the similarity threshold
            if correlation >= threshold:
                similar_images.append((test_image_name, test_image))

    return similar_images

def find_similar_images(query_image_path, similar_images_paths=None, threshold=0.8, refine_results=True):
    """
    Find and return similar images to the query image based on color histogram comparison.
    """
    query_image = cv2.imread(query_image_path)
    if query_image is None:
        print(f"ERROR: Unable to read image '{query_image_path}'.")
        return []

    query_hist = calculate_color_histogram(query_image)

    similar_images = []

    if refine_results and similar_images_paths is not None:
        # Apply similar image search on the CBIR results
        for test_image_url in similar_images_paths:
            response = requests.get(test_image_url)
            test_image = np.asarray(bytearray(response.content), dtype="uint8")
            test_image = cv2.imdecode(test_image, cv2.IMREAD_COLOR)

            test_hist = calculate_color_histogram(test_image)

            correlation = calculate_correlation(query_hist, test_hist)

            if correlation >= threshold:
                similar_images.append((test_image_url, test_image))
                #print(f"Similar image found: {test_image_url}")
                #print(f"Correlation: {correlation}")

    else:
        # Apply similar image search on all images in the database
        test_images_folder = os.path.dirname(query_image_path)

        for test_image_name in os.listdir(test_images_folder):
            test_image_path = os.path.join(test_images_folder, test_image_name)
            if test_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image = cv2.imread(test_image_path)
                if test_image is None:
                    print(f"ERROR: Unable to read image '{test_image_path}'.")
                    continue

                test_hist = calculate_color_histogram(test_image)

                correlation = calculate_correlation(query_hist, test_hist)

                if correlation >= threshold:
                    # Construct the URL of the similar image
                    image_url = url_to_local_path(test_image_path, settings.MEDIA_ROOT)
                    similar_images.append((image_url, test_image))
                    #print(f"Similar image found: {image_url}")
                    #print(f"Correlation: {correlation}")
    """
    if similar_images:
        print("Similar images have been found.")
    else:
        print("No similar images were found.")
    """

    return similar_images
