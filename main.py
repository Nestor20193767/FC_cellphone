import streamlit as st
import cv2
import numpy as np
import tempfile
import plotly.graph_objects as go


# Función para aplicar un filtro FIR a la señal
def apply_fir_filter(signal, fs, lowcut=2.2, highcut=3.2, numtaps=101):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Crear los coeficientes del filtro utilizando una ventana de Hamming
    taps = np.hamming(numtaps)

    # Normalizar para obtener la respuesta deseada
    taps *= (high - low)

    # Hacer el sinc y la convolución
    sinc_low = np.sinc(2 * low * (np.arange(numtaps) - (numtaps - 1) / 2.))
    sinc_high = np.sinc(2 * high * (np.arange(numtaps) - (numtaps - 1) / 2.))

    taps = sinc_high - sinc_low

    # Aplicar la ventana de Hamming
    taps *= np.hamming(numtaps)

    # Normalizar los coeficientes del filtro
    taps /= np.sum(taps)

    # Filtrar la señal
    filtered_signal = np.convolve(signal, taps, mode='same')

    return filtered_signal


# Función para realizar el detrending utilizando una media móvil
def detrend(signal, window_size=81):
    # Crear una ventana de promediado
    window = np.ones(window_size) / window_size

    # Convolucionar la señal con la ventana para obtener la tendencia
    trend = np.convolve(signal, window, mode='same')

    # Restar la tendencia de la señal original para obtener la señal detrended
    detrended_signal = signal - trend

    return detrended_signal

# Función para procesar el video y calcular los valores del canal rojo
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

    hb = (fc / 60) * 28

    ps = 60*pow(b/(hb-a),0.5)
    pd = c * fc + d

    return ps, pd


# Función para estimar PAM (Presión Arterial Media) a partir de PS y PD
def estimate_pam(ps, pd):
    return pd + (1 / 3) * (ps - pd)


# Título de la aplicación
st.title('Blood Pressure by Camera')

# Subir video
video_file = st.file_uploader("Subir video", type=["mp4", "avi", "mov"])

# Selector de sexo (Male/Female)
MorF = st.selectbox('Selecciona el sexo', ['Male', 'Female'], key='maleorfemale')

# Botón para comenzar el procesamiento
start = st.button('Start')
st.write('Esta pagina fue creada para probar la funcionalidad de la app CardioCare+ ')
# Dividir la pantalla en dos columnas
col1, col2 = st.columns(2)

# Columna 1: Mostrar video
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

        # Aplicar filtro FIR a los valores del canal rojo
        #filtered_red_values = red_values
        filtered_red_values = apply_fir_filter(red_values, fps)
        filtered_red_values = detrend(filtered_red_values)

        # Calcular frecuencia cardíaca
        heart_rate, peaks = calculate_heart_rate(filtered_red_values, fps)

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
        fig.add_trace(go.Scatter(x=np.arange(len(filtered_red_values)), y=filtered_red_values, mode='lines',
                                 name='Valores del canal rojo filtrados'))
        fig.add_trace(go.Scatter(x=peaks, y=np.array(filtered_red_values)[peaks], mode='markers',
                                 marker=dict(symbol='x', size=10), name='Picos'))
        fig.update_layout(title='Valores del canal rojo y Picos', xaxis_title='Tiempo', yaxis_title='Valor',
                          template='plotly_white')
        st.plotly_chart(fig)

