import argparse
import cv2 as cv


def generate_mask(image_path):
    img = cv.imread(image_path)

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

    return thresh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="dataset root path")
    args = parser.parse_args()

    dataset_root = args.path

    # TODO: Parse input folder and save generated masks