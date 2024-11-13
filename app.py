import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Funzione per analizzare l'immagine
def analyze_image(image):
    # Converte l'immagine in formato OpenCV
    image = np.array(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Definisce i range di colore per il verde e il rosso
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def filter_contours(contours, min_area=100):
        return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    contours_green = filter_contours(contours_green)
    contours_red = filter_contours(contours_red)

    rectangles = []

    for cnt in contours_green:
        x, y, w, h = cv2.boundingRect(cnt)
        rectangles.append((x, 'V'))

    for cnt in contours_red:
        x, y, w, h = cv2.boundingRect(cnt)
        rectangles.append((x, 'R'))

    rectangles.sort(key=lambda rect: rect[0])
    sequence = ''.join([rect[1] for rect in rectangles])
    return sequence

# Interfaccia Streamlit
st.title("Analizzatore di Immagini per Rettangoli Colorati")
uploaded_file = st.file_uploader("Carica un'immagine", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Immagine Caricata", use_column_width=True)

    # Analizza l'immagine
    sequence = analyze_image(image)
    st.write("Sequenza rilevata:")
    st.code(sequence)