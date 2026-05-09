import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# run: python src/proyecto_pds.py

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