import cv2
import numpy as np
import os

# suppose your OCR result array is stored in `result`
# result = np.array([...])  # [[[x y] ... ]]

# load original image
def padddlecrop(img_path):
    import paddleocr
    from paddleocr import TextDetection
    ocr_model = TextDetection()  # Set lang to 'en' for English
    result = ocr_model.predict(img_path)
    result1=result[0]["dt_polys"]
    polygons = [[[np.int32(x), np.int32(y)] for (x, y) in poly] for poly in result1]
    img = cv2.imread(img_path)

    # output folder
    out_dir = "crops"
    os.makedirs(out_dir, exist_ok=True)

    # loop through each polygon
    for i, pts in enumerate(result1):
        pts = np.array(pts, dtype=np.int32)

        # get bounding rectangle
        x, y, w, h = cv2.boundingRect(pts)

        # crop
        crop = img[y:y+h, x:x+w]

        # save as PNG
        filename = os.path.join(out_dir, f"crop_{i+1}.png")
        cv2.imwrite(filename, crop)

    return polygons







