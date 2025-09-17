import easyocr
import cv2
import numpy as np
import os
def ocr_and_crop(img_path):
    reader1 = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    reader2 = easyocr.Reader(['hi'])
    reader3 = easyocr.Reader(['ch_tra'])
    reader4 = easyocr.Reader(['ar'])

    result1 = reader1.readtext(img_path)  # your OCR results
    result2 = reader2.readtext(img_path)  # your OCR results 
    result3 = reader3.readtext(img_path)
    result4 = reader4.readtext(img_path)

    english_text = [text for (_, text, _) in result1]
    hindi_text = [text for (_, text, _) in result2]
    chinese_text = [text for (_, text, _) in result3]
    arabic_text = [text for (_, text, _) in result4]



    # your OCR results


    # load original image
    img = cv2.imread(img_path)

    hindi_confidence = [conf for (_, _, conf) in result2]
    chinese_confidence = [conf for (_, _, conf) in result3]
    english_confidence = [conf for (_, _, conf) in result1]
    arabic_confidence = [conf for (_, _, conf) in result4]

    lang_list = []
    bounding_boxes = [bbox for (bbox, _, _) in result1]
    for i in range(len(hindi_confidence)):
        if hindi_confidence[i] > english_confidence[i] and hindi_confidence[i] > chinese_confidence[i] and hindi_confidence[i] > arabic_confidence[i]:
            lang_list.append('hi')
        elif chinese_confidence[i] > english_confidence[i] and chinese_confidence[i] > hindi_confidence[i] and chinese_confidence[i] > arabic_confidence[i]:
            lang_list.append('ch_tra')
        elif arabic_confidence[i] > english_confidence[i] and arabic_confidence[i] > hindi_confidence[i] and arabic_confidence[i] > chinese_confidence[i]:
            lang_list.append('ar')
        else:
            lang_list.append('en')
    # print(lang_list)
    out_dir = "D:\OCR\deeptextrecognitionbenchmark\crops"
    os.makedirs(out_dir, exist_ok=True)

    # loop through results
    for i, (bbox, text, score) in enumerate(result1):
        # convert to numpy array
        pts = np.array(bbox, dtype=np.int32)

        # get bounding rectangle
        x, y, w, h = cv2.boundingRect(pts)

        # crop region
        crop = img[y:y+h, x:x+w]

        # save crop as PNG
        filename = os.path.join(out_dir, f"crop_{i+1}.png")
        cv2.imwrite(filename, crop)
    return lang_list,bounding_boxes,english_text,hindi_text,chinese_text,arabic_text

def crop(img_path):
    reader1 = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    result1 = reader1.readtext(img_path)  # your OCR results
    bounding_boxes = [bbox for (bbox, _, _) in result1]
    text = [text for (_, text, _) in result1]

    # load original image
    img = cv2.imread(img_path)

    out_dir = "D:\OCR\deeptextrecognitionbenchmark\crops"
    os.makedirs(out_dir, exist_ok=True)

    # loop through results
    for i, (bbox, text, score) in enumerate(result1):
        # convert to numpy array
        pts = np.array(bbox, dtype=np.int32)

        # get bounding rectangle
        x, y, w, h = cv2.boundingRect(pts)

        # crop region
        crop = img[y:y+h, x:x+w]

        # save crop as PNG
        filename = os.path.join(out_dir, f"crop_{i+1}.png")
        cv2.imwrite(filename, crop)

        return bounding_boxes
    
def ind_lang(img_path,lang):
    reader1 = easyocr.Reader(['lang']) # this needs to run only once to load the model into memory
    result1 = reader1.readtext(img_path)  # your OCR results
    bounding_boxes = [bbox for (bbox, _, _) in result1]

    # load original image
    img = cv2.imread(img_path)

    out_dir = r"D:\OCR\deeptextrecognitionbenchmark\crops"
    os.makedirs(out_dir, exist_ok=True)

    # loop through results
    for i, (bbox, text, score) in enumerate(result1):
        # convert to numpy array
        pts = np.array(bbox, dtype=np.int32)

        # get bounding rectangle
        x, y, w, h = cv2.boundingRect(pts)

        # crop region
        crop = img[y:y+h, x:x+w]

        # save crop as PNG
        filename = os.path.join(out_dir, f"crop_{i+1}.png")
        cv2.imwrite(filename, crop)

        return bounding_boxes,text




