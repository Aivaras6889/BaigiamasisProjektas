import os
from pathlib import Path
import numpy as np
import skimage.filters as filters
import cv2

"""Uzkrovimo funkcija v.0"""
# def load_images(src,extensions):
#     """
#     loaded - list of total images loaded 
#     src
    
    
#     """
#     img = []
#     src_objects = []
#     matching_paths = []

#     """Loading file paths"""

#     for ext in extensions:
#         for path_obj in src.rglob(f"*{ext}"):
#             matching_paths.append(path_obj)

#     """Sorting file paths"""
#     matching_paths.sort()

#     for path_obj in matching_paths:
#         if path_obj.is_file():
#             img = cv2.imread(str(path_obj))
#             if img is not None:
#                 img.append(img)
#                 src_objects.append(path_obj)
#             else:
#                 print(f"  Warning: Could not load image {path_obj.name}")
#         else:
#             print(f"  Skipping non-file item: {path_obj.name}")
    
#     return img




"""Uzkrovimo funkcija V1"""

def load_images(src: Path, extensions: tuple) -> list:
    """
    Loads images recursively from a source directory.

    Args:
        src (pathlib.Path): The root directory as a Path object.
        extensions (tuple): A tuple of file extensions (e.g., ('.png', '.jpg')).

    Returns:
        list: A list of OpenCV image objects (NumPy arrays) successfully loaded.
    """
    src_obj = Path(src)
    loaded_images = []  # This list will store all loaded image data
    matching_paths = [] # Temporary list for collecting all file paths

    # Collect all matching file paths recursively
    for ext in extensions:
        for path_obj in src_obj.rglob(f"*{ext}"):
            if path_obj.is_file():
                matching_paths.append(path_obj)

    # Sort paths for consistent order (optional, but good practice)
    matching_paths.sort()

    # Load images from the collected paths
    for path_obj in matching_paths:
        image = cv2.imread(str(path_obj)) # Load the individual image

        if image is not None:
            loaded_images.append(image) # Add the loaded image to our list
        else:
            print(f"Warning: Could not load image {path_obj.name}")
    
    return loaded_images


# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv

# def readTrafficSigns(rootpath):
#     '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

#     Arguments: path to the traffic sign data, for example './GTSRB/Training'
#     Returns:   list of images, list of corresponding labels'''
#     images = [] # images
#     labels = [] # corresponding labels
#     # loop over all 42 classes
#     for c in range(0,43):
#         prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
#         gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
#         gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
#         next(gtReader, None)
#         for row in gtReader:
#             images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
#             labels.append(row[7]) # the 8th column is the label
#         gtFile.close()
#     return images, labels



def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels, list of file paths'''
    images = []  # List to store images
    labels = []  # List to store corresponding labels
    image_paths = []  # List to store corresponding file paths
    
    # Loop over all 43 classes (0 to 42)
    for c in range(0, 43):
        # Construct subdirectory for class c
        prefix = os.path.join(rootpath, format(c, '05d')) + '/'
        
        # Open the annotations file for this class
        gtFile = open(os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv'))
        
        # Use CSV reader to parse the annotations file
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader, None)  # Skip the header row
        
        # Read the image and label data
        for row in gtReader:
            # Read the image using cv2.imread (ensure it's in BGR format)
            img_path = os.path.join(prefix, row[0])  # Full image path
            img = cv2.imread(img_path)
            
            # Check if the image is loaded correctly
            if img is not None:
                images.append(img)
                labels.append(row[7])  # The 8th column in CSV is the label
                image_paths.append(img_path)  # Store image path for saving later
            else:
                print(f"Could not read image: {img_path}")
        
        gtFile.close()
    
    return images, labels, image_paths


