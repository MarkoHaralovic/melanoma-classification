import argparse
import cv2 as cv
import os
from tqdm import tqdm
import concurrent.futures
import multiprocessing

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

def process_image(args):
    image_path, mask_path = args
    if os.path.exists(mask_path):
        return False
    
    try:
        mask = generate_mask(image_path)
        cv.imwrite(mask_path, mask)
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", help="dataset root path")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of parallel workers (default: number of CPU cores)")
    
    args = parser.parse_args()

    dataset_root = args.source_path
    num_workers = args.num_workers or max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_workers} workes")
    os.makedirs(os.path.join(dataset_root, "masks"), exist_ok=True)
    
    tasks = []
    for folder in os.listdir(dataset_root):
        if folder in ["train", "val", "test"]:
            os.makedirs(os.path.join(dataset_root, "masks", folder), exist_ok=True)
            for subfolder in os.listdir(os.path.join(dataset_root, folder)):
                os.makedirs(os.path.join(dataset_root, "masks", folder, subfolder), exist_ok=True)
                for image_file in os.listdir(os.path.join(dataset_root, folder, subfolder)):
                    image_path = os.path.join(dataset_root, folder, subfolder, image_file)
                    mask_path = os.path.join(dataset_root, "masks", folder, subfolder, image_file.replace(".jpg", "_mask.jpg"))
                    tasks.append((image_path, mask_path))
    
    processed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Generating masks"))
        processed_count = sum(results)
    
    print(f"Processed {processed_count} new images. {len(tasks) - processed_count} were already processed or failed.")