import os
import cv2
import numpy as np
import skimage.filters as filters
from image_utils import readTrafficSigns


# def images_preprocessing(images):
#     edited_images=[]
#     for image in images:
#     # Convert to grayscale
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Resize image
#         image_resized = cv2.resize(image_gray, (64, 64), interpolation=cv2.INTER_AREA)
        
#         # Histogram equalization
#         image_equalized = cv2.equalizeHist(image_resized)
        
#         # Apply Gaussian blur
#         image_blur = cv2.GaussianBlur(image_equalized, (7, 7), 0)
        
#         # Sharpen image using unsharp mask (output is a float image)
#         image_sharped = filters.unsharp_mask(image_blur, radius=5, amount=3)*255
        
#         # Clip the image values to be within 0-255 and convert to uint8
#         # image_sharped = np.uint8(np.clip(image_sharped * 255, 0, 255))  # Ensure the range is 0-255
#         edited_images.append(image_resized)
#         # Append to edited images list
        
#     return edited_images

# def images_preprocessing_debug(images):
#     """Preprocessing with debugging information"""
#     from skimage import filters
#     import cv2
#     import numpy as np
    
#     edited_images = []
    
#     for i, image in enumerate(images):
#         if i == 0:  # Debug first image
#             print(f"Original image shape: {image.shape}, dtype: {image.dtype}")
        
#         # Convert to grayscale
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         if i == 0:
#             print(f"After grayscale: {image_gray.shape}, dtype: {image_gray.dtype}")
        
#         # Resize image
#         image_resized = cv2.resize(image_gray, (64, 64), interpolation=cv2.INTER_AREA)
#         if i == 0:
#             print(f"After resize: {image_resized.shape}, dtype: {image_resized.dtype}")
        
#         # Histogram equalization
#         image_equalized = cv2.equalizeHist(image_resized)
#         if i == 0:
#             print(f"After equalization: {image_equalized.shape}, dtype: {image_equalized.dtype}")
        
#         # Apply Gaussian blur
#         image_blur = cv2.GaussianBlur(image_equalized, (7, 7), 0)
#         if i == 0:
#             print(f"After blur: {image_blur.shape}, dtype: {image_blur.dtype}")
        
#         # Sharpen image using unsharp mask
#         image_sharped = filters.unsharp_mask(image_blur, radius=5, amount=3)
#         if i == 0:
#             print(f"After unsharp: {image_sharped.shape}, dtype: {image_sharped.dtype}, range: {image_sharped.min():.3f} to {image_sharped.max():.3f}")
        
#         # Convert back to uint8
#         image_final = np.uint8(np.clip(image_sharped * 255, 0, 255))
#         if i == 0:
#             print(f"Final: {image_final.shape}, dtype: {image_final.dtype}, range: {image_final.min()} to {image_final.max()}")
        
#         edited_images.append(image_final)
    
#     return edited_images

def safe_preprocessing(images):
    import cv2
    import numpy as np
    from skimage import filters

    edited_images = []
    for i, image in enumerate(images):
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image

            # Resize
            image_resized = cv2.resize(image_gray, (96, 96))

            # Histogram equalization
            image_equalized = cv2.equalizeHist(image_resized)

            # Optional: Gaussian blur
            image_blur = cv2.GaussianBlur(image_equalized, (3, 3), 0)

            # Optional: Sharpen
            image_sharp = filters.unsharp_mask(image_blur, radius=2, amount=1.5) * 255
            image_final = np.uint8(np.clip(image_sharp, 0, 255))

            edited_images.append(image_final)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    return edited_images


# images,labels = readTrafficSigns(src)

def read_images_list(image_paths):
    """Simply read images from a list of paths"""
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Could not read: {path}")
    
    return images

def get_labels_from_csv(csv_file_path):
    import pandas as pd
    df = pd.read_csv(csv_file_path, sep=';')  # Specify the separator
    print("CSV columns:", df.columns)
    print(df.head())
    return df['ClassId'].values  # Now this will work


# def process_and_save_images(images, image_paths):
#     edited_images = []
    
#     # Process each image
#     for image, path in zip(images, image_paths):
#         # Grayscale conversion
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Resize image
#         image_resized = cv2.resize(image_gray, (128, 128), interpolation=cv2.INTER_AREA)
        
#         # Histogram equalization
#         image_equalized = cv2.equalizeHist(image_resized)
        
#         # Apply Gaussian blur
#         image_blur = cv2.GaussianBlur(image_equalized, (7, 7), 0)
        
#         # Sharpen image using unsharp mask
#         image_sharped = filters.unsharp_mask(image_blur, radius=10, amount=7) * 125
        
#         # Add the processed image to the list
#         edited_images.append(image_sharped)
        
#         # Get the folder from the image path
#         subfolders = os.listdir(path)  # Get the subfolder from the path
        
#         # Ensure the subfolder exists, create it if it doesn't
#         if not os.path.exists(subfolders):
#             os.makedirs(subfolders)  # Create subfolder if it doesn't exist
        
#         # Save the edited image back to the original location
#         image_to_save = np.uint8(np.clip(image_sharped, 0, 255))  # Ensure the image is in valid range
#         cv2.imwrite(path, image_to_save)  # Save the image
        
#         print(f"Processed and saved: {path}")
    
#     return edited_images


def get_images_and_labels(data_folder_path):
    dirs = os.listdir(data_folder_path)
    suffix = ('.txt', '.csv')  # Files to exclude
    dirs_list = [dir for dir in dirs if not dir.endswith(suffix)]  # Only process valid directories
    labels = []  # List to hold labels (directory names)
    image_paths = []  # List to hold image file paths
    images = []  # List to hold loaded image data

    # Loop through each valid directory
    for dir_name in dirs_list:
        subject_dir_path = os.path.join(data_folder_path, dir_name)

        # Skip if it's not a directory
        if not os.path.isdir(subject_dir_path):
            continue

        # Loop through each file in the subdirectory
        for image_name in os.listdir(subject_dir_path):
            image_path = os.path.join(subject_dir_path, image_name)

            # Skip .csv files and non-image files
            if image_name.endswith(suffix):
                continue
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.ppm')):
                continue  # Skip non-image files

            # Read the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                # print(f"Warning: Could not read image: {image_path}")  # Log issues with reading images
                continue  # Skip invalid images

            # Debug: Check if image is None or not
            # print(f"Loaded image: {image_path} with shape {image.shape}")

            # Append image and corresponding label to the lists
            images.append(image)
            
            labels.append(dir_name)  # The directory name is used as the label
    print(len(images))
    print(len(labels))
    return labels, images

def get_images(image_paths):
    imagest=[]
    for image_path in image_paths:
        imaget=cv2.imread(image_path)
        imagest.append(imaget)
    return imagest

def get_image_paths(data_folder_path):
    dirs = os.listdir(data_folder_path)
    image_paths = []
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

    # Check files directly in the root folder
    for file_name in dirs:
        file_path = os.path.join(data_folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().split('.')[-1] in allowed_extensions:
            image_paths.append(file_path)

    # Check subdirectories
    for dir_name in dirs:
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        if not os.path.isdir(subject_dir_path):
            continue
        for file_name in os.listdir(subject_dir_path):
            file_path = os.path.join(subject_dir_path, file_name)
            if file_name.lower().split('.')[-1] in allowed_extensions:
                image_paths.append(file_path)
            else:
                print(f"Skipping non-image file: {file_path}")

    return image_paths



def analyze_dataset(labels, images):
    """Analyze your dataset to find the problem"""
    from collections import Counter
    
    print("=== DATASET ANALYSIS ===")
    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")
    
    # Check unique classes
    unique_labels = set(labels)
    print(f"Unique classes: {len(unique_labels)}")
    print(f"Ratio (unique/total): {len(unique_labels)/len(labels):.3f}")
    
    # Show class distribution
    label_counts = Counter(labels)
    print(f"\nClass distribution (top 10):")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count} images")
    
    print(f"\nClass distribution (bottom 10):")
    for label, count in label_counts.most_common()[-10:]:
        print(f"  {label}: {count} images")
    
    # Check for problems
    if len(unique_labels) > len(labels) * 0.5:
        print("\n⚠️  WARNING: Too many unique classes!")
        print("   This will cause ML algorithm errors")
    
    # Check for single-sample classes
    single_sample_classes = [label for label, count in label_counts.items() if count == 1]
    if single_sample_classes:
        print(f"\n⚠️  WARNING: {len(single_sample_classes)} classes have only 1 sample!")
        print("   Examples:", single_sample_classes[:5])
    
    return label_counts


def show_images_within_range(images,value):
    for i, image in enumerate(images):
        if i > value:
            break
        cv2.imshow(f'Image {i+5}',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows

def clean_labels(labels):
    return [str(int(label)) for label in labels]

# def save_training_data(data_folder_path,images):
#     # List subdirectories (each representing a different person)
#     dirs = os.listdir(data_folder_path)

   
#     for dir_name in dirs:
#         subject_dir_path = os.path.join(data_folder_path, dir_name)
#         if not os.path.isdir(subject_dir_path):
#             continue

#         for image_name in os.listdir(subject_dir_path):
#             image_path = os.path.join(subject_dir_path, image_name)
#             if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.ppm')):
#                 print(f"Skipping non-image file: {image_name}")
#                 continue  # Skip non-image files like CSVs
            
            
#             image = cv2.imread(images)
#             if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
#                 success = cv2.imwrite(image_path, image)
#             else:
#                 print(f"Invalid image data at: {image_path}")
        
#             # Check if the image was saved successfully
#             if success:
#                 print(f"Image saved successfully: {image_path}")
#             else:
#                 print(f"Failed to save image: {image_path}")

# def save_and_return_edited_images(edited_images, image_paths):
#     updated_images = []  # List to store the saved (updated) images

#     for edited_image, image_path in zip(edited_images, image_paths):
        
#         # Ensure the directory exists before saving
#         subfolder = os.path.dirname(image_path)
#         if not os.path.exists(subfolder):
#             os.makedirs(subfolder)  # Create the subfolder if it doesn't exist
        
    

#         # Ensure the image is in a valid format for saving (values between 0 and 255)
#         image_to_save = np.uint8(np.clip(edited_image, 0, 255))
        
#         # Save the processed image back to its original location
#         try:
#             success = cv2.imwrite(image_path, image_to_save)
#         except Exception as e:
#             print(e)

#         if success:
#             print(f"Image saved successfully: {image_path}")
#             updated_images.append(edited_image)  # Add to the list of updated images
#         else:
#             print(f"Failed to save image: {image_path}")
    
#     return updated_images  # Return the list of processed and saved images

# def preprocessing(images):

# Example usage
train_src = "app/data/dataset/train"
# images, labels, image_paths = readTrafficSigns(src)
labels_train, images_train = get_images_and_labels(data_folder_path=train_src)
# print(len(images))
# image_paths = get_image_paths(data_folder_path=src)
# print(len(image_paths))
test_src = "app/data/dataset/test"
# labels_test, images_test = get_images_and_labels(data_folder_path=test_src)
images_paths =  get_image_paths(test_src)
images_test= get_images(images_paths)
labels_src = "app\data\dataset\GT-final_test.csv"
labels_test = get_labels_from_csv(labels_src)

# images_test= get_images(images_path)
# print(len(imagest))


train_edited_images = safe_preprocessing(images_train)
test_edited_images = safe_preprocessing(images_test)
# print(len(train_edited_images))
# print(len(test_edited_images))
# print(len(labels_test))





# edited_images = []
# for image in images:
#     # Convert to grayscale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Resize image
#     image_resized = cv2.resize(image_gray, (128, 128), interpolation=cv2.INTER_AREA)
    
#     # Histogram equalization
#     image_equalized = cv2.equalizeHist(image_resized)
    
#     # Apply Gaussian blur
#     image_blur = cv2.GaussianBlur(image_equalized, (7, 7), 0)
    
#     # Sharpen image using unsharp mask (output is a float image)
#     image_sharped = filters.unsharp_mask(image_blur, radius=7, amount=4)*255
    
#     # Clip the image values to be within 0-255 and convert to uint8
#     # image_sharped = np.uint8(np.clip(image_sharped * 255, 0, 255))  # Ensure the range is 0-255
    
#     # Append to edited images list
#     edited_images.append(image_sharped)
# # updated_images = save_and_return_edited_images( edited_images=edited_images,image_paths=image_paths)
# for label in labels:
#     print(type(labels))
train_labels_clean = clean_labels(labels_train)
test_labels_clean = clean_labels(labels_test)

# Naudojimas:
# labels, images = get_images_and_labels(src)
analysis_train = analyze_dataset(labels_train, train_edited_images)
analysis_test = analyze_dataset(labels_test, test_edited_images)


# print(cleaned_labels)



# show_images_within_range(train_edited_images,5)
    


# print(cleaned_labels)
# for i, image in enumerate(edited_images):
#     if i >= 5:  # Only show first 5 images
#         break
#     cv2.imshow(f'Image {i+1}', image)  # Need window name and image
#     cv2.waitKey(0)
#     cv2.destroyAllWindows() 


