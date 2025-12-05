# Modelo de Hindmarsh-Rose con Echo State Network (ESN)

## 📋 Descripción

Modelo para predicción de dinámicas caóticas del sistema de Hindmarsh-Rose usando **Echo State Network (ESN)** basado en el **Teorema de Takens**.

## 🎯 Características del Sistema

- **Tipo:** Sistema caótico de 3 dimensiones (neuronal)
- **Variable observable:** x (potencial de membrana)
- **Comportamiento:** Spikes irregulares (bursting caótico)
- **Desafío:** Horizonte de predicción limitado por exponente de Lyapunov

## 📁 Archivos

- `generar_datos_hindmarsh_rose.py` - Genera serie temporal caótica y guarda en CSV
- `modelo_hindmarsh_rose_esn.py` - Modelo ESN con entrenamiento y evaluación
- `comparar_modelos.py` - Comparación visual entre modelo físico y ESN
- `ejecutar_pipeline.py` - Script para ejecutar todo el pipeline automáticamente

## 🚀 Ejecución Rápida

### Con UV (Recomendado)

```bash
# Pipeline completo
uv run --no-project ejecutar_pipeline.py

# O paso a paso
uv run --no-project generar_datos_hindmarsh_rose.py
uv run --no-project modelo_hindmarsh_rose_esn.py
uv run --no-project comparar_modelos.py
```

### Con Python tradicional

```bash
# Pipeline completo
python ejecutar_pipeline.py

# O paso a paso
python generar_datos_hindmarsh_rose.py
python modelo_hindmarsh_rose_esn.py
python comparar_modelos.py
```

## 📊 Salidas Generadas

- `datos_hindmarsh_rose.csv` - Serie temporal caótica
- `datos_hindmarsh_rose_visualizacion.png` - Visualización de datos (4 gráficas)
- `modelo_esn_hindmarsh_rose.npz` - Modelo ESN entrenado (pesos)
- `esn_hindmarsh_rose_resultados.png` - Resultados de evaluación (4 gráficas):
  - Predicción one-step
  - Predicción multi-step con horizonte de Lyapunov
  - Error relativo
  - Atractor 3D
- `comparacion_hindmarsh_rose.png` - Comparación modelo físico vs ESN (2×2 grid):
  - Serie completa: observado vs físico vs ESN
  - Test set one-step (primeros 500 puntos)
  - Predicción multi-step autónoma (200 pasos)
  - Distribución del error absoluto

## 🏗️ Arquitectura ESN

- **Tipo de red:** Echo State Network (Reservoir Computing)
- **Reservoir size:** 300 neuronas
- **Spectral radius:** 0.9 (edge of chaos)
- **Input scale:** 1.0
- **Entrenamiento:** Ridge Regression (solución cerrada)

## 📈 Resultados Esperados

### One-Step Prediction
- **RMSE:** < 10% del rango (aceptable para caos)
- **MAE:** Error promedio absoluto bajo

### Multi-Step Prediction
- **Horizonte de Lyapunov:** > 5 pasos (criterio de éxito)
- **Error estadístico:** Media y desviación estándar dentro de ±15%

## 🔬 Diferencias con el Modelo de Cloro

| Aspecto | Cloro (LSTM) | Hindmarsh-Rose (ESN) |
|---------|--------------|---------------------|
| **Comportamiento** | Suave, predecible | Caótico, impredecible |
| **Arquitectura** | LSTM (2 capas) | ESN (reservorio fijo) |
| **Entrenamiento** | Backpropagation | Ridge Regression |
| **Métrica clave** | RMSE < 5% | Horizonte Lyapunov > 5 |
| **Predicción** | Many-step precisa | Multi-step limitada |
| **Tiempo** | Minutos | Segundos |

## ✅ Ventajas de ESN para Caos

1. **Rapidez:** Entrenamiento en segundos (no requiere backpropagation)
2. **Memoria dinámica:** El reservorio mantiene historia del sistema
3. **Edge of chaos:** Spectral radius ≈ 1 maximiza capacidad computacional
4. **Robusto:** Menos propenso a overfitting que redes profundas

## 📚 Ver documentación completa

Consulta el [README principal](../README.md) y la [guía técnica](../documentos/guia_tecnica_reglas.md) para:
- Fundamento teórico completo
- Justificación de hiperparámetros
- Comparación con otros métodos
- Interpretación de resultados
