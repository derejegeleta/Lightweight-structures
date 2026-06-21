import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from matplotlib import pyplot as plt

def apply_morphological_operations(image, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated  = cv2.dilate(image, kernel, iterations=1)
    eroded   = cv2.erode(image, kernel, iterations=1)
    opened   = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed   = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    tophat   = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return dilated, eroded, opened, closed, gradient, tophat, blackhat

def run_parallel(images, kernel_size=3):
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(apply_morphological_operations, img, kernel_size) for img in images]
        results = [f.result() for f in futures]
    return results

def main():
    # --- Load and preprocess image ---
    img = cv2.imread('cv.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image 'cv.png' not found. Place it in the same folder.")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # --- Benchmark single-core ---
    start = time.time()
    dilated, eroded, opened, closed, gradient, tophat, blackhat = apply_morphological_operations(binary)
    end = time.time()
    print(f"Single-core execution time: {end - start:.6f} seconds")

    # --- Benchmark multi-core (simulate batch of images) ---
    images = [binary for _ in range(8)]  # replicate image for parallel test
    start = time.time()
    parallel_results = run_parallel(images)
    end = time.time()
    print(f"Multi-core execution time (8 images): {end - start:.6f} seconds")

    # --- Visualization ---
    titles = ['Original Image','Binary Mask','Dilated','Eroded','Opened','Closed','Gradient','Top-Hat','Black-Hat']
    images_show = [img, binary, dilated, eroded, opened, closed, gradient, tophat, blackhat]

    plt.figure(figsize=(14,8))
    for i in range(len(images_show)):
        plt.subplot(3,3,i+1)
        plt.imshow(images_show[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
