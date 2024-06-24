import streamlit as st
import cv2
import numpy as np
import tempfile
import plotly.graph_objects as go
import plotly.express as px

# Función para procesar el video y calcular los valores del canal rojo y la frecuencia cardíaca
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

# Función para encontrar los picos en la señal de valores del canal rojo
def find_peaks(signal, threshold=1.0):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append(i)
    return np.array(peaks)

# Función para calcular la frecuencia cardíaca a partir de los valores del canal rojo y la frecuencia de fotogramas
def calculate_heart_rate(red_values, fps):
    threshold = np.mean(red_values)
    peaks = find_peaks(red_values, threshold)
    heart_rate = len(peaks) / (len(red_values) / fps) * 60
    return heart_rate, peaks

# Función para estimar PS (Presión Sistólica) y PD (Presión Diastólica) basado en la frecuencia cardíaca y sexo
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

# Función para estimar PAM (Presión Arterial Media) a partir de PS y PD
def estimate_pam(ps, pd):
    return pd + (1/3) * (ps - pd)

# Título de la aplicación
st.title('Blood Pressure by Camera')

# Subir video
video_file = st.file_uploader("Subir video", type=["mp4", "avi", "mov"])

# Selector de sexo (Male/Female)
MorF = st.selectbox('Selecciona el sexo', ['Male', 'Female'], key='maleorfemale')

# Botón para comenzar el procesamiento
start = st.button('Start')

# Dividir la pantalla en dos columnas
col1, col2 = st.columns(2)

# Columna 1: Visualización en tiempo real del video
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

# Columna 2: Procesamiento del video y visualización de resultados
with col2:
    if start and video_file is not None:
        # Procesar el video para obtener los valores del canal rojo y la frecuencia de fotogramas
        red_values, fps = process_video(tfile.name)

        # Calcular frecuencia cardíaca
        heart_rate, peaks = calculate_heart_rate(red_values, fps)

        # Estimar PS y PD según el sexo seleccionado
        sex = 'male' if MorF == 'Male' else 'female'
        estimated_ps, estimated_pd = estimate_ps_pd(heart_rate, sex)

        # Estimar PAM a partir de PS y PD
        estimated_pam = estimate_pam(estimated_ps, estimated_pd)

        # Mostrar resultados calculados
        st.write(f'Frecuencia cardíaca estimada: {heart_rate:.2f} latidos por minuto')
        st.write(f'Presión sistólica estimada (PS): {estimated_ps:.2f} mmHg')
        st.write(f'Presión diastólica estimada (PD): {estimated_pd:.2f} mmHg')
        st.write(f'Presión arterial media estimada (PAM): {estimated_pam:.2f} mmHg')

        # Crear la gráfica con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(red_values)), y=red_values, mode='lines', name='Valores del canal rojo'))
        fig.add_trace(go.Scatter(x=peaks, y=np.array(red_values)[peaks], mode='markers', marker=dict(symbol='x', size=10), name='Picos'))
        fig.update_layout(title='Valores del canal rojo y Picos', xaxis_title='Tiempo', yaxis_title='Valor', template='plotly_white')
        st.plotly_chart(fig)

