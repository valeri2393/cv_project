import streamlit as st
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
import cv2
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

weights = {
    'Axial': 'models/besaxial_40epoch.pt',
    'Coronal': 'models/best_coronal.pt',
    'Sagittal': 'models/sagittal_best50epoch.pt'
}

@st.cache_resource
def load_all_models():
    models = {}

    model_ax = torch.hub.load('ultralytics/yolov5', 'custom', path=weights['Axial'], force_reload=True, device='cpu')
    model_ax.eval()
    models['Axial'] = model_ax

    model_cor = torch.hub.load('ultralytics/yolov5', 'custom', path=weights['Coronal'], force_reload=True, device='cpu')
    model_cor.eval()
    models['Coronal'] = model_cor

    model_sag = torch.hub.load('ultralytics/yolov5', 'custom', path=weights['Sagittal'], force_reload=True, device='cpu')
    model_sag.eval()
    models['Sagittal'] = model_sag

    return models

models = load_all_models()

st.title('Определение опухолей с помощью моделей YOLA')

option = st.selectbox('Выбери тип среза:', ('Axial', 'Coronal', 'Sagittal'))
model = models[option]

confidence_threshold = st.slider('Установите порог уверенности:', 0.0, 1.0, 0.5, 0.01)

def predict(image, model, conf_thres):
    img = np.array(image)
    results = model(img)
    results = results.xyxy[0][results.xyxy[0][:, 4] > conf_thres]
    return results

def display_results(image, results):
    img = np.array(image)
    if len(results) > 0:
        for *xyxy, conf, cls in results.cpu().numpy():
            label = 'tumor' if int(cls) == 0 else 'background'
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        label = 'background'
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def load_image(image):
    if isinstance(image, BytesIO):
        return Image.open(image)
    else:
        return load_image_from_url(image)

uploaded_files = st.file_uploader("Выберите изображения...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        results = predict(image, model, confidence_threshold)
        result_img = display_results(image, results)
        
        st.image(result_img, caption=f"Результат предсказания ({option})", use_column_width=True)

image_urls = st.text_area('Введите URL изображений (один URL на строку)', height=100).strip().split('\n')
image_urls = [url.strip() for url in image_urls if url.strip()]

if image_urls:
    for url in image_urls:
        image = load_image_from_url(url).convert("RGB")
        
        results = predict(image, model, confidence_threshold)
        result_img = display_results(image, results)
        
        st.image(result_img, caption=f"Результат предсказания ({option})", use_column_width=True)
