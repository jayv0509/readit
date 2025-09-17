import google.generativeai as genai

def lang_based_correction(text):
    """Corrects text based on language using a generative AI model.
     Args:
         text (str): The input text to be corrected.
         Returns:
         tuple: A tuple containing two lists - the corrected text lines and their corresponding language codes.
         """
    genai.configure(api_key="AIzaSyDLPkWjQ90owvgC0vXei6RbkngsYWgsdfc")
    client = genai.GenerativeModel()
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""

Do not change the order of words or the overall sentence structure.  
Correct spelling mistakes and fix word spacing so the sentence becomes meaningful.   
If the text is in multiple languages, write one complete language"s text in one line and another language's text in another line. 
Also, identify the language of each line in iso 639 codes like en,hi,es etc.

Format the output exactly like this:

text = {"""line1,
line2,
line3,
line4"""}

lang_list = {"line1_lang, line2_lang, line3_lang, line4_lang"}

Text to correct:

{text}



    """

    response = model.generate_content(prompt)
    s = response.text
    s = response.text
    lines = s.strip().splitlines()

    text_part = []
    lang_part = []

    # Flag to know if we are reading text or lang_list
    reading_text = False
    reading_lang = False

    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        
        if line.startswith("text ="):
            reading_text = True
            reading_lang = False
            # remove "text =" then split if comma exists
            cleaned = line.replace("text =", "", 1).strip().rstrip(",")
            if cleaned:
                text_part.append(cleaned)
        elif line.startswith("lang_list ="):
            reading_lang = True
            reading_text = False
            cleaned = line.replace("lang_list =", "", 1).strip()
            if cleaned:
                lang_part.extend([l.strip() for l in cleaned.split(",") if l.strip()])
        else:
            # continuation line
            if reading_text:
                text_part.append(line.rstrip(","))
            elif reading_lang:
                lang_part.extend([l.strip() for l in line.split(",") if l.strip()])
    return text_part,lang_part

def correct_ocr_text(text):
    """Corrects OCR text using a generative AI model.
    Args:
        text (str): The OCR text to be corrected.
        Returns:
        str: The corrected text.
        """
    genai.configure(api_key="AIzaSyDLPkWjQ90owvgC0vXei6RbkngsYWgsdfc")
    client = genai.GenerativeModel()
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""

Strictly do not change the order of words or the overall text structure.What word is in the line should stay in that line only.  
Correct spelling mistakes and fix word spacing so the sentence becomes meaningful.   
If the text is in multiple languages, write each corrected text without changing the word order and in its own line. 






Text to correct:

{text}



    """

    response = model.generate_content(prompt)
    return response.text
    

def prediction_of_lang(text_list):
    """Predicts the language of each line of text using a generative AI model.And also corrects the spellings.
    Args:
        text_list (list): A list of text lines.
        Returns:
        tuple: A tuple containing two lists - the corrected text lines and their corresponding language codes.
        """
    text = "\n".join(str(item) for item in text_list)

    genai.configure(api_key="AIzaSyDLPkWjQ90owvgC0vXei6RbkngsYWgsdfc")
    client = genai.GenerativeModel()
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
Fix the spellings in each line and give the lines in the same order as they are given.
identify the language of each line in iso 639 codes like en,hi,es etc.
Strictly maintain the number of lines and their order.
Format the output exactly like this:
text =  "line1,line2,line3,line4...
lang = "line1_lang, line2_lang, line3_lang,line4_lang...."








Text to correct:

{text}



#     """
    response = model.generate_content(prompt)
    s = response.text
    lines = s.splitlines()

    text_line = lines[0].split("=", 1)[1].strip().strip('"')
    lang_line = lines[1].split("=", 1)[1].strip().strip('"')

    text_list = [t.strip() for t in text_line.split(",")]
    lang_list = [l.strip() for l in lang_line.split(",")]


    return text_list,lang_list







    
    








