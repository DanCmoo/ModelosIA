# Modelo de Predicción de Concentración de Cloro con LSTM

## 📋 Descripción

Modelo de predicción de concentración de cloro usando **LSTM (Long Short-Term Memory)** basado en el **Teorema de Takens** para reconstrucción del espacio de fases.

## 📁 Archivos

- `generar_datos_cloro.py` - Genera la serie temporal de cloro y guarda en CSV
- `modelo_cloro_lstm.py` - Modelo LSTM principal con entrenamiento y evaluación
- `ejecutar_pipeline.py` - Script para ejecutar todo el pipeline automáticamente
- `comparar_modelos.py` - Compara modelo físico original con predicciones LSTM

## 🚀 Ejecución Rápida

### Con UV (Recomendado)

```bash
# Pipeline completo
uv run --no-project ejecutar_pipeline.py

# O paso a paso
uv run --no-project generar_datos_cloro.py
uv run --no-project modelo_cloro_lstm.py

# Comparar modelo físico vs LSTM
uv run --no-project comparar_modelos.py
```

### Con Python tradicional

```bash
# Pipeline completo
python ejecutar_pipeline.py

# O paso a paso
python generar_datos_cloro.py
python modelo_cloro_lstm.py
```

## 📊 Salidas Generadas

- `datos_cloro.csv` - Serie temporal de concentración
- `datos_cloro_visualizacion.png` - Gráficas de los datos generados
- `modelo_lstm_cloro.h5` - Modelo entrenado
- `lstm_cloro_resultados.png` - Gráficas de evaluación del modelo
- `comparacion_modelo_fisico_vs_lstm.png` - Comparación visual detallada (6 gráficas):
  - Serie temporal completa
  - Comparación en test set
  - Zoom detallado
  - Error de predicción
  - Correlación observado vs predicho
  - Distribución de valores

## 🎯 Arquitectura

- **Tipo de red:** LSTM (Long Short-Term Memory)
- **Capas:** 7 capas (Dense + LSTM + Dropout)
- **Hidden units:** 64
- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE

## 📈 Resultados Esperados

- **RMSE:** < 5% del rango
- **MAE:** < 3% del rango
- **Horizonte de predicción:** Predicción one-step-ahead precisa

## 📚 Ver documentación completa

Consulta el [README principal](../README.md) para más detalles sobre:
- Fundamento teórico (Teorema de Takens)
- Instalación de dependencias
- Explicación detallada del proceso
- Interpretación de resultados
