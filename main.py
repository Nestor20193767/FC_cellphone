import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    red_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extraer los valores de color rojo (puedes ajustar según el color de tu piel)
        red_value = np.mean(frame[:, :, 2])  # Canal rojo
        red_values.append(red_value)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return red_values, fps

def find_peaks(signal, threshold=1.0):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append(i)
    return np.array(peaks)

def calculate_heart_rate(red_values, fps):
    threshold = np.mean(red_values)
    peaks = find_peaks(red_values, threshold)
    heart_rate = len(peaks) / (len(red_values) / fps) * 60
    return heart_rate, peaks

def estimate_ps_pd(fc, sex='male'):
    if sex == 'male':
        a, b = 0.5, 90  # Coeficientes para hombres
        c, d = 0.3, 60
    else:
        a, b = 0.4, 85  # Coeficientes para mujeres
        c, d = 0.25, 55

    ps = a * fc + b
    pd = c * fc + d

    return ps, pd

def estimate_pam(ps, pd):
    return pd + (1/3) * (ps - pd)

st.title('Blood Pressure by Camera')


# Subir video
video_file = st.file_uploader("Subir video", type=["mp4", "avi", "mov"])

MorF = st.selectbox('Selecciona el sexo', ['Male', 'Female'], key='maleorfemale')

start = st.button('Start')

col1, col2 = st.columns(2)

with col1:

    if start:
        if video_file is not None:
            # Guardar el video subido en un archivo temporal
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            # Usar OpenCV para leer el video
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB")

            cap.release()

with col2:
    if start and video_file is not None:
        # Procesar el video para obtener los valores del canal rojo
        red_values, fps = process_video(tfile.name)

        # Calcular frecuencia cardíaca
        heart_rate, peaks = calculate_heart_rate(red_values, fps)

        # Estimar PS y PD
        sex = 'male' if MorF == 'Male' else 'female'
        estimated_ps, estimated_pd = estimate_ps_pd(heart_rate, sex)

        # Estimar PAM
        estimated_pam = estimate_pam(estimated_ps, estimated_pd)

        st.write(f'Frecuencia cardíaca estimada: {heart_rate:.2f} latidos por minuto')
        st.write(f'Presión sistólica estimada (PS): {estimated_ps:.2f} mmHg')
        st.write(f'Presión diastólica estimada (PD): {estimated_pd:.2f} mmHg')
        st.write(f'Presión arterial media estimada (PAM): {estimated_pam:.2f} mmHg')

        # Crear la gráfica
        fig, ax = plt.subplots()
        ax.plot(red_values, label='Valores del canal rojo')
        ax.plot(peaks, np.array(red_values)[peaks], 'x', label='Picos')
        ax.grid()
        ax.legend()

        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)
