from demo import text_recognition
from language_identification import language_identify
from translation import translate_text_sync
from llm import prediction_of_lang
from llm import lang_based_correction
from llm import correct_ocr_text
from makingjson import makejson
image_path = input("enter the image path : ")
lang = input("enter the language of your text in iso 639 format like en,hi,es etc. and for multiple language model you can leave it blank,we also have handwritten hindi model,for it enter ha_hi:")
bounding_boxes,lang_list,pred_list,ordered_text = text_recognition(image_path,lang)
corrected_text_list,lang_list_from_llm = prediction_of_lang(pred_list)
final_lang_list = language_identify(lang_list_from_llm,lang_list)
output_method = input(print("How do you want your output?\nWe have 3 options as listed:\n1.THe layout of text in image\n2.Json format with coordinates and langauge of each detected text\n3.Each line with its langauge sepparated by comma\nEnter 1 or 2 or 3"))
if output_method == '1':
    final_text = correct_ocr_text(ordered_text)
    print(final_text)
elif output_method == '2':
    makejson(bounding_boxes,final_lang_list,corrected_text_list)
elif output_method == '3':
    new_text_list , new_lang_list = lang_based_correction(ordered_text)
    for text,lang in zip(new_text_list,new_lang_list):
        print(f"{text} : {lang}")
lang = input("Do you want to translate the text? y/n : ")
if lang == 'y':
    target_lang = input("Enter the target language in iso 639 format like en,hi,es etc. : ")
    if output_method == '3':
        for text,lang in zip(new_text_list,new_lang_list):
            translated_text = translate_text_sync(text,lang,target_lang)
            print(f"Original text: {text} ({lang}) -> Translated text: {translated_text} ({target_lang})")
    else:
        new_text_list , new_lang_list = lang_based_correction(ordered_text)
        for text,lang in zip(new_text_list,new_lang_list):
            translated_text = translate_text_sync(text,lang,target_lang)
            print(f"Original text: {text} ({lang}) -> Translated text: {translated_text} ({target_lang})")


    





