# PolyOCR  

PolyOCR is a **multilingual Optical Character Recognition (OCR)** system that works across multiple languages, scripts, and even handwritten text. It supports both **individual language models** (where you specify the target language) and a **poly-model** that automatically handles **multi-language images** without requiring any prior language specification.  

---

## 🌍 Supported Languages  

'en', 'de', 'es', 'fr', 'it', 'nl', 'pl', 'pt',
'sw', 'tr', 'vi', 'zh', 'hi', 'mr', 'ha_hi', 'ar'


- **Handwritten Hindi**: A special fine-tuned model for handwritten Hindi text.  
- **Poly-model**: Automatically recognizes multiple languages in the same image.  

---

## ✨ Key Features  

- **Flexible Models**  
  - Use **single-language models** by specifying the language.  
  - Use the **poly-model** for multi-language images (no prior language input needed).  

- **Handwritten OCR**  
  - Fine-tuned model for handwritten Hindi.  
  - Works across fonts, handwriting styles, and mixed-language text.  

- **Multiple Output Formats**  
  1. Text with **layout preserved** (same structure as the image).  
  2. **JSON file** with:
     - Detected text  
     - Bounding box coordinates  
     - Language of each region  
  3. Text **grouped by language** (e.g., all Hindi text, all English text, etc.).  

- **Translation Support**  
  - Translate recognized text into any target language using `googletrans`.  

---

## 🔄 Processing Pipeline  

For the **poly-model**, the pipeline is:  

1. **Text Detection** → [CRAFT (EasyOCR)]( )  
2. **Cropping** → via OpenCV  
3. **Text Recognition** → Language-based recognition models:  
   - Latin  
   - Devanagari  
   - Chinese  
   - Arabic  
   (via EasyOCR)  
4. **Language Recognition** → Confidence scores from recognition models  
5. **Output Generation** → `{text list, language list, bounding boxes}`  
6. **Post-processing** → LLM (Gemini) for:  
   - Fixing spelling errors  
   - Splitting joined words  
7. **Script-to-Language Disambiguation** → LLM (Gemini) refines results:  
   - Devanagari → Hindi / Marathi  
   - Latin → English / Spanish / Italian  
8. **Translation Module** → `googletrans` for multilingual translation.  

---

## 🛠 Fine-tuning  

- 3 models were finetuned on datasets containing signboards and one on handwritten dataset as you can see in [demo.py](D:\OCR\deeptextrecognitionbenchmark\demo.py) . Fine-tuning was done using the [deep-text-recognition-benchmark]( ) repo.  

### 📚 Datasets  

- **Latin** → [IIIT5k]( https://r.search.yahoo.com/_ylt=AwrPrgcB88poPAIAT5y7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Ny/RV=2/RE=1759340546/RO=10/RU=https%3a%2f%2fcvit.iiit.ac.in%2fresearch%2fprojects%2fcvit-projects%2fthe-iiit-5k-word-dataset/RK=2/RS=8VnktUCKU29N1.bl1EGVwuCD8TY-)  
- **Devanagari** → [IIIT-IndicSTR-Word](https://cvit.iiit.ac.in/usodi/istr.php )  
- **Chinese** → Real-world dataset [TC-STR 7k-word](https://r.search.yahoo.com/_ylt=AwrKBEXd88poWgIAX0a7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Ny/RV=2/RE=1759340765/RO=10/RU=https%3a%2f%2fgithub.com%2fesun-ai%2ftraditional-chinese-text-recogn-dataset/RK=2/RS=ThIDDLUOvesh20rMZzZctTroC88-) + **120k synthetic images** from [synthtiger](https://github.com/clovaai/synthtiger)  
- **Handwritten Hindi** → [IIIT-Indic-HW-Words](https://ilocr.iiit.ac.in/dataset/18/)  

---

## 📊 Accuracy Comparison  

| Model              | Previous Accuracy | Fine-tuned Accuracy |
|--------------------|-------------------|----------------------|
| Chinese            | 57.758%           | **81.685%**         |
| Hindi              | 23.460%           | **82.260%**         |
| English            | 75.450%           | **81.200%**         |
| Handwritten Hindi  | 5.229%            | **68.464%**         |  

> ⚠️ Fine-tuned models (`latin.pth`, `devnagari.pth`, `chinese.pth`) are **not used in the default pipeline** to keep PolyOCR more general-purpose.  
They can be enabled by uncommenting code in `demo.py`, but require the CRAFT detector to also be fine-tuned for **word-level detection**.  

---

## 📂 Pre-trained Models  

Links to previous models used for fine-tuning:  

- [Latin Model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/latin.zip)  
- [Devanagari Model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip )  
- [Chinese Model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese.zip )

You will find more detail about this models in [Easyocr repo](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py)
