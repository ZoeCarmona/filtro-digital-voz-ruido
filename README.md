# Proyecto Final de Procesamiento Digital de Señales

## Filtrado digital de una señal de voz con ruido

Este repositorio contiene el desarrollo del proyecto final de la materia **Procesamiento Digital de Señales**, en el cual se trabaja con una señal real de voz, se le agrega ruido de forma controlada y posteriormente se diseña un filtro digital para mejorar la calidad de la señal.

El objetivo principal del proyecto es aplicar conceptos de procesamiento digital de señales como análisis en el dominio del tiempo, análisis espectral mediante la Transformada Rápida de Fourier (FFT), diseño de filtros digitales y evaluación del desempeño mediante métricas de error.

---

## Descripción general

El proyecto consiste en grabar una señal de voz real, agregarle ruido blanco para simular una condición de interferencia y analizar cómo cambia la señal tanto en el dominio del tiempo como en el dominio de la frecuencia.

Posteriormente, se diseña y aplica un filtro digital con el propósito de atenuar el ruido y conservar, en la mayor medida posible, la inteligibilidad de la voz.

---

## Objetivos

- Manipular señales de audio en formato `.wav` usando Python.
- Graficar señales discretas en el dominio del tiempo.
- Generar y sumar ruido blanco a una señal de voz.
- Analizar el espectro de una señal mediante FFT.
- Diseñar y aplicar un filtro digital FIR o IIR.
- Comparar la señal original, la señal ruidosa y la señal filtrada.
- Evaluar el desempeño del filtrado mediante métricas como ECM y SNR.

---

## Estructura del proyecto

```text
proyecto-final-pds-filtrado-voz/
│
├── audios/
│   ├── voz_original.wav
│   ├── ruido_blanco.wav
│   ├── voz_ruidosa_A.wav
│   ├── voz_ruidosa_B.wav
│   └── voz_filtrada.wav
│
├── graficas/
│   ├── voz_original_completa.png
│   ├── voz_original_zoom.png
│   ├── voz_ruidosa.png
│   ├── espectro_ruidosa.png
│   ├── voz_filtrada.png
│   └── espectro_filtrada.png
│
├── src/
│   └── proyecto_pds.py
│
├── reporte/
│   └── reporte_final.pdf
│
└── README.md
