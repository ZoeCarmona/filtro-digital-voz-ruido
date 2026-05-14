import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# run: python src/proyecto_pds.py
# source venv/Scripts/activate
# python -m pip install numpy scipy matplotlib

# Crear carpetas si no existen
os.makedirs("graficas", exist_ok=True)
os.makedirs("audios_generados", exist_ok=True)

# ==============================
# Parte 1: Cargar señal de voz
# ==============================

fs, x = wavfile.read("audios/voz_original.wav")

# Si la señal está en estéreo, convertir a mono
if len(x.shape) > 1:
    x = x.mean(axis=1)

# Convertir a float
x = x.astype(float)

# Normalizar para que esté entre -1 y 1
x = x / np.max(np.abs(x))

# Vector de tiempo
N = len(x)
t = np.arange(N) / fs

print("Frecuencia de muestreo:", fs, "Hz")
print("Número de muestras:", N)
print("Duración:", N / fs, "segundos")

# ==============================
# Gráfica de señal completa
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Señal original de voz en el dominio del tiempo")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_original_completa.png", dpi=300)
plt.show()

# ==============================
# Zoom de 50 a 100 ms
# ==============================

inicio = 0.05
fin = 0.10

idx_inicio = int(inicio * fs)
idx_fin = int(fin * fs)

plt.figure(figsize=(10, 4))
plt.plot(t[idx_inicio:idx_fin], x[idx_inicio:idx_fin])
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Zoom de la señal original de voz: 50 a 100 ms")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_original_zoom.png", dpi=300)
plt.show()

# ==============================
# Parte 2: Generación y suma de ruido blanco
# ==============================

from scipy.io.wavfile import write

# Fijar semilla para que el ruido generado sea reproducible
np.random.seed(42)

# Generar ruido blanco con el mismo tamaño que la señal original
ruido = np.random.normal(0, 1, N)

# Normalizar el ruido
ruido = ruido / np.max(np.abs(ruido))

# Factores de ruido
alpha_A = 0.08  # ruido moderado
alpha_B = 0.30  # ruido fuerte

# Crear señales ruidosas
x_ruidosa_A = x + alpha_A * ruido
x_ruidosa_B = x + alpha_B * ruido

# Evitar saturación dejando las señales entre -1 y 1
x_ruidosa_A = x_ruidosa_A / np.max(np.abs(x_ruidosa_A))
x_ruidosa_B = x_ruidosa_B / np.max(np.abs(x_ruidosa_B))

# Guardar audios generados
write("audios_generados/voz_ruidosa_A.wav", fs, (x_ruidosa_A * 32767).astype(np.int16))
write("audios_generados/voz_ruidosa_B.wav", fs, (x_ruidosa_B * 32767).astype(np.int16))

print("Audios ruidosos generados correctamente.")

# ==============================
# Gráfica de señal ruidosa A completa
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t, x_ruidosa_A)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Señal de voz con ruido moderado en el dominio del tiempo")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_ruidosa_A_completa.png", dpi=300)
plt.show()

# ==============================
# Zoom de señal ruidosa A
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t[idx_inicio:idx_fin], x_ruidosa_A[idx_inicio:idx_fin])
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Zoom de señal de voz con ruido moderado: 50 a 100 ms")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_ruidosa_A_zoom.png", dpi=300)
plt.show()

# ==============================
# Gráfica de señal ruidosa B completa
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t, x_ruidosa_B)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Señal de voz con ruido fuerte en el dominio del tiempo")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_ruidosa_B_completa.png", dpi=300)
plt.show()

# ==============================
# Zoom de señal ruidosa B
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t[idx_inicio:idx_fin], x_ruidosa_B[idx_inicio:idx_fin])
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Zoom de señal de voz con ruido fuerte: 50 a 100 ms")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_ruidosa_B_zoom.png", dpi=300)
plt.show()

# ==============================
# Parte 3: Análisis espectral con FFT
# ==============================

def graficar_espectro(senal, fs, titulo, nombre_archivo):
    """
    Calcula y grafica el espectro de magnitud de una señal usando FFT.
    Se usa escala en decibeles para observar mejor las diferencias.
    """
    N = len(senal)

    # Calcular FFT
    X = np.fft.fft(senal)

    # Frecuencias
    frecuencias = np.fft.fftfreq(N, d=1/fs)

    # Nos quedamos con la mitad positiva
    mitad = N // 2
    frecuencias_positivas = frecuencias[:mitad]

    # Magnitud
    magnitud = np.abs(X[:mitad])

    # Convertir a decibeles
    magnitud_db = 20 * np.log10(magnitud + 1e-12)

    # Graficar
    plt.figure(figsize=(10, 4))
    plt.plot(frecuencias_positivas, magnitud_db)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title(titulo)
    plt.grid(True)
    plt.xlim(0, 8000)
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.show()

# Espectro de la señal original
graficar_espectro(
    x,
    fs,
    "Espectro en dB de la señal de voz original",
    "graficas/espectro_db_voz_original.png"
)

# Espectro de la señal con ruido moderado
graficar_espectro(
    x_ruidosa_A,
    fs,
    "Espectro en dB de la señal de voz con ruido moderado",
    "graficas/espectro_db_voz_ruidosa_A.png"
)

# Espectro de la señal con ruido fuerte
graficar_espectro(
    x_ruidosa_B,
    fs,
    "Espectro en dB de la señal de voz con ruido fuerte",
    "graficas/espectro_db_voz_ruidosa_B.png"
)

print("Análisis espectral realizado correctamente.")

# ==============================
# Parte 4: Diseño y aplicación del filtro digital
# ==============================

from scipy.signal import firwin, lfilter, freqz

# Parámetros del filtro
frecuencia_corte = 3500  # Hz
orden_filtro = 101       # número de coeficientes del filtro FIR

# Diseñar filtro FIR pasa-bajas con ventana Hamming
coeficientes_filtro = firwin(
    numtaps=orden_filtro,
    cutoff=frecuencia_corte,
    fs=fs,
    window="hamming",
    pass_zero="lowpass"
)

# Aplicar filtro a la señal con ruido fuerte
y_filtrada = lfilter(coeficientes_filtro, 1.0, x_ruidosa_B)

# Normalizar señal filtrada
y_filtrada = y_filtrada / np.max(np.abs(y_filtrada))

# Guardar audio filtrado
write("audios_generados/voz_filtrada.wav", fs, (y_filtrada * 32767).astype(np.int16))

print("Filtro digital aplicado correctamente.")
print("Tipo de filtro: FIR pasa-bajas")
print("Frecuencia de corte:", frecuencia_corte, "Hz")
print("Orden del filtro:", orden_filtro)

# ==============================
# Respuesta en frecuencia del filtro
# ==============================

w, h = freqz(coeficientes_filtro, worN=8000, fs=fs)

plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.title("Respuesta en frecuencia del filtro FIR pasa-bajas")
plt.grid(True)
plt.xlim(0, 8000)
plt.tight_layout()
plt.savefig("graficas/respuesta_filtro_pasabajas.png", dpi=300)
plt.show()

# ==============================
# Señal filtrada completa
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t, y_filtrada)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Señal de voz filtrada en el dominio del tiempo")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_filtrada_completa.png", dpi=300)
plt.show()

# ==============================
# Zoom de la señal filtrada
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t[idx_inicio:idx_fin], y_filtrada[idx_inicio:idx_fin])
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Zoom de la señal de voz filtrada: 50 a 100 ms")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/voz_filtrada_zoom.png", dpi=300)
plt.show()

# ==============================
# Comparación entre señal ruidosa B y filtrada
# ==============================

plt.figure(figsize=(10, 4))
plt.plot(t, x_ruidosa_B, label="Señal con ruido fuerte", alpha=0.7)
plt.plot(t, y_filtrada, label="Señal filtrada", alpha=0.7)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.title("Comparación entre señal ruidosa y señal filtrada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/comparacion_ruidosa_filtrada.png", dpi=300)
plt.show()

# ==============================
# Espectro de la señal filtrada
# ==============================

graficar_espectro(
    y_filtrada,
    fs,
    "Espectro en dB de la señal de voz filtrada",
    "graficas/espectro_db_voz_filtrada.png"
)

# ==============================
# Parte 5: Evaluación del desempeño y métricas de error
# ==============================

def calcular_ecm(senal_original, senal_comparada):
    """
    Calcula el Error Cuadrático Medio entre dos señales.
    """
    return np.mean((senal_original - senal_comparada) ** 2)


def calcular_snr(senal_original, senal_comparada):
    """
    Calcula la Relación Señal-Ruido en dB.
    """
    ruido_error = senal_original - senal_comparada
    potencia_senal = np.mean(senal_original ** 2)
    potencia_ruido = np.mean(ruido_error ** 2)

    return 10 * np.log10(potencia_senal / potencia_ruido)


# Como el filtro FIR introduce un pequeño retardo, se compensa para comparar mejor
retardo = (orden_filtro - 1) // 2

x_comp = x[:-retardo]
x_ruidosa_B_comp = x_ruidosa_B[:-retardo]
y_filtrada_comp = y_filtrada[retardo:]

# Calcular ECM
ecm_ruidosa = calcular_ecm(x_comp, x_ruidosa_B_comp)
ecm_filtrada = calcular_ecm(x_comp, y_filtrada_comp)

# Calcular SNR
snr_antes = calcular_snr(x_comp, x_ruidosa_B_comp)
snr_despues = calcular_snr(x_comp, y_filtrada_comp)

print("\nMétricas de desempeño")
print("----------------------")
print("ECM señal ruidosa B vs original:", ecm_ruidosa)
print("ECM señal filtrada vs original:", ecm_filtrada)
print("SNR antes del filtrado:", snr_antes, "dB")
print("SNR después del filtrado:", snr_despues, "dB")

# ==============================
# Gráfica comparativa de métricas ECM
# ==============================

plt.figure(figsize=(8, 4))
plt.bar(["Ruidosa B", "Filtrada"], [ecm_ruidosa, ecm_filtrada])
plt.ylabel("ECM")
plt.title("Comparación del Error Cuadrático Medio")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("graficas/comparacion_ecm.png", dpi=300)
plt.show()

# ==============================
# Gráfica comparativa de SNR
# ==============================

plt.figure(figsize=(8, 4))
plt.bar(["Antes del filtrado", "Después del filtrado"], [snr_antes, snr_despues])
plt.ylabel("SNR [dB]")
plt.title("Comparación de SNR antes y después del filtrado")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("graficas/comparacion_snr.png", dpi=300)
plt.show()