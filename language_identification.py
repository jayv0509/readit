# Use a pipeline as a high-level helper
# -*- coding: utf-8 -*-
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# def language_detection(text):
#     MODEL = "papluca/xlm-roberta-base-language-detection"   # or another model from HF Hub
#     device = 0 if torch.cuda.is_available() else -1

#     # load tokenizer + model
#     tokenizer = AutoTokenizer.from_pretrained(MODEL)

#     model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#     # create a simple pipeline for inference
#     lang_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

#     model.to(device)
#     # inference (single string)
#     result = lang_pipe(text)
#     lang = result[0]['label']   # -> [{'label': 'hi', 'score': 0.98}]
#     print(lang)
#     if lang == 'hi':
#         model_name = "GautamDaksh/Native_Marathi_Hindi_English_classifier"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

#         def classify_sentence(sentence):
#             words = sentence.split()
#             predictions = []
#             label_map_inv = {0: "E", 1: "H", 2: "M"}  # Reverse mapping for labels
#             h = 0  # Hindi count
#             m = 0  # Marathi count  

#             for word in words: 

#                 inputs = tokenizer(word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
#                 outputs = model(**inputs)
#                 predicted_class = torch.argmax(outputs.logits, dim=-1).item()
#                 if predicted_class == 1:
#                     h += 1
#                 elif predicted_class == 2:
#                     m += 1
#                 predictions.append((word, label_map_inv[predicted_class]))

#             if h > m:
#                 refined_lang = 'hi'
#             elif m > h: 
#                 refined_lang = 'mr'
#             return predictions, refined_lang

#         predictions, refined_lang = classify_sentence(text) 
#     else:
#         refined_lang = lang
#     return refined_lang
def language_identify(refined_lang,lang_list):
    refined_lang_list = []
    
    for language,lang in zip(lang_list,refined_lang):
        if language == "en" and lang in ['en','de', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'sw', 'tr', 'vi']:
            refined_lang_list.append(lang)
        elif language == "hi" and lang in ['hi','mr']:
            refined_lang_list.append(lang)
        elif language == "ch_tra":
            refined_lang_list.append(language)
        elif language == "ar":
            refined_lang_list.append(language)
        else:
            refined_lang_list.append(language)
        
    return refined_lang_list
