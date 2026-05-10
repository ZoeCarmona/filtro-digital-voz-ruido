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