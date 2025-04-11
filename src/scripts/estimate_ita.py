import argparse
import csv
import os
import warnings

import cv2 as cv
import numpy as np
import kneed
from tqdm import tqdm

CSV_HEADER = ["image", "ita", "color_r", "color_g", "color_b"]


def get_ita_angle(color_rgb: np.ndarray) -> float:
    color_lab = cv.cvtColor(np.uint8([[color_rgb]]), cv.COLOR_RGB2LAB)[0][0]
    return np.arctan((color_lab[0] - 50) / color_lab[2]) * 180 / np.pi


def kmeans_dominant_color_lab(processed_img, k):
    processed_img_lab = cv.cvtColor(processed_img, cv.COLOR_BGR2LAB)
    pixel_values = processed_img_lab.reshape((-1, 3))
    # remove black pixels
    pixel_values = pixel_values[np.where(pixel_values[:, 0] > 0)]
    # keep only a and b channels
    # pixel_values = pixel_values[:, 1:]
    pixel_values = np.float32(pixel_values)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, (centers) = cv.kmeans(
        pixel_values, k, None, criteria, 10, cv.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    labels = labels.flatten()

    dominant_label = np.argmax(np.bincount(labels))
    dominant_color = centers[dominant_label]
    dominant_color = np.array(dominant_color)
    dominant_color = np.round(dominant_color).astype(int)
    # add back L channel
    dominant_color = cv.cvtColor(np.uint8([[dominant_color]]), cv.COLOR_LAB2RGB)
    return dominant_color, compactness


def kmeans_dominant_color(image_path):
    img = cv.imread(image_path)
    img = cv.resize(img, (128, 256))
    # img = cv.resize(img, (256, 512))
    # img = cv.resize(img, (512, 1024))

    # Isolate skin
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    lab_planes = list(cv.split(lab))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv.merge(lab_planes)
    clahe_img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    ## Dullrazor
    grayscale = cv.cvtColor(clahe_img, cv.COLOR_BGR2GRAY)  # [1] uses RGB2GRAY
    kernel = cv.getStructuringElement(1, (9, 9))  # [1] uses a 3x3 kernel
    blackhat = cv.morphologyEx(grayscale, cv.MORPH_BLACKHAT, kernel)
    blurred = cv.GaussianBlur(blackhat, (3, 3), cv.BORDER_DEFAULT)

    _, hair_mask = cv.threshold(
        blurred, 20, 255, cv.THRESH_BINARY
    )  # [2] sets the threshold at 10, [1] at 25
    masked_img = cv.bitwise_and(clahe_img, clahe_img, mask=255 - hair_mask)

    ## Threshold to remove pigmentations
    hsv = cv.cvtColor(masked_img, cv.COLOR_BGR2HSV)
    _, _, v = cv.split(hsv)
    v = cv.GaussianBlur(v, (5, 5), 0)
    _, v_thresh = cv.threshold(v, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thresh = v_thresh

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=5)
    thresh = cv.dilate(thresh, kernel, iterations=5)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=5)

    thresh = cv.bitwise_or(thresh, hair_mask)
    final_image = cv.bitwise_and(img, img, mask=255 - thresh)

    ks = range(3, 10)
    colors, compactnesses = zip(
        *[kmeans_dominant_color_lab(final_image, k) for k in ks]
    )
    kneedle = kneed.KneeLocator(
        ks, compactnesses, S=1.0, curve="convex", direction="decreasing"
    )

    if kneedle.elbow is None:
        dominant_color = colors[-1]
    else:
        dominant_color = colors[kneedle.elbow]
    return dominant_color


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="dataset root path")
    parser.add_argument("-o", "--out", help="output csv file")
    parser.add_argument("-f", "--files", nargs="+", help="list of filenames to process")
    args = parser.parse_args()

    dataset_root = args.path
    csv_path = args.out if args.out != None else "skin_tones.csv"
    files = args.files if args.files != None else os.listdir(dataset_root)

    if os.path.exists(csv_path):
        raise FileExistsError(f"File {csv_path} already exists")

    with open(csv_path, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for filename in tqdm(sorted(files)):
            file_path = os.path.join(dataset_root, filename)

            if not os.path.exists(file_path):
                warnings.warn(f"Skipping file {file_path}, file does not exist")
                continue

            try:
                color = kmeans_dominant_color(file_path).squeeze()
                angle = get_ita_angle(color)

                writer.writerow([filename, angle, color[0], color[1], color[2]])
            except Exception as e:
                warnings.warn(f"ITA estimation failed on file {filename}: {e}")
