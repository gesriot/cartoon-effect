import os
import sys
from PIL import Image, ImageOps
import numpy as np
import cv2
from sklearn.cluster import KMeans


def apply_cartoon_effect(image, median_blur_ksize=5, gaussian_blur_ksize=5, adaptive_thresh_c=10):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, median_blur_ksize)
    edges = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (gaussian_blur_ksize, gaussian_blur_ksize), 0), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, adaptive_thresh_c)
    color = cv2.bilateralFilter(img_array, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

def cartoon_effect_brown_lines_and_sepia(image, median_blur_ksize=5, gaussian_blur_ksize=5, adaptive_thresh_c=10):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, median_blur_ksize)
    edges = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (gaussian_blur_ksize, gaussian_blur_ksize), 0), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, adaptive_thresh_c)
    color = cv2.bilateralFilter(img_array, 9, 300, 300)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_colored[np.where((edges_colored == [255, 255, 255]).all(axis=2))] = [255, 255, 255]
    edges_colored[np.where((edges_colored == [0, 0, 0]).all(axis=2))] = [139, 69, 19]
    cartoon = cv2.bitwise_and(color, edges_colored)

    gray = cv2.cvtColor(cartoon, cv2.COLOR_RGB2GRAY)
    recolored = np.stack([gray, gray, gray], axis=-1)
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    recolored = recolored @ sepia_matrix.T
    recolored = np.clip(recolored, 0, 255).astype(np.uint8)
    return Image.fromarray(recolored)

def reduce_colors(image, num_colors):
    data = np.array(image)
    pixels = data.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    new_data = new_pixels.reshape(data.shape).astype('uint8')
    new_image = Image.fromarray(new_data)
    return new_image

def replace_colors(image):
    data = np.array(image)
    
    lightest_color = [255, 255, 255]  # Белый
    medium_color = [255, 237, 216]    # Светло-коричневый
    darkest_color = [182, 148, 130]   # Темно-коричневый
    
    unique_colors = np.unique(data.reshape(-1, data.shape[2]), axis=0)
    
    brightness = np.dot(unique_colors, [0.299, 0.587, 0.114])
    
    sorted_colors = unique_colors[np.argsort(brightness)]
    
    data[(data == sorted_colors[0]).all(axis=2)] = darkest_color
    data[(data == sorted_colors[-1]).all(axis=2)] = lightest_color
    for color in sorted_colors[1:-1]:
        data[(data == color).all(axis=2)] = medium_color
    
    new_image = Image.fromarray(data)
    return new_image

def process_image(image_path, output_folder):
    image = Image.open(image_path)
    cartoon_image = apply_cartoon_effect(image)
    cartoon_sepia_image = cartoon_effect_brown_lines_and_sepia(cartoon_image)
    final_image = reduce_colors(cartoon_sepia_image, 3)
    final_image = replace_colors(final_image)
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '.png')
    final_image.save(output_path)

def process_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder)

def main():
    if len(sys.argv) < 2:
        print("Введите путь до изображения или до папки с изображениями.")
        return

    input_path = sys.argv[1]
    output_folder = "processed"

    if len(sys.argv) > 2:
        output_folder = sys.argv[2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isfile(input_path):
        process_image(input_path, output_folder)
    elif os.path.isdir(input_path):
        process_folder(input_path, output_folder)
    else:
        print("Неверный путь.")

if __name__ == '__main__':
    main()
