# Modelo de Predicción de Concentración de Cloro con LSTM

## 📋 Descripción

Este proyecto implementa un modelo de predicción de concentración de cloro usando **LSTM (Long Short-Term Memory)** basado en el **Teorema de Takens** para reconstrucción del espacio de fases. El proyecto sigue estrictamente las reglas y especificaciones de la guía técnica.

## 🎯 Objetivos

- Predecir la concentración de cloro en un sistema de tratamiento de agua
- Demostrar que la IA puede aprender dinámicas sin conocer las ecuaciones subyacentes
- Cumplir con todas las reglas de implementación (G1-G4, T1-T3, L1-L5, V1-V2)

## 🔬 Fundamento Teórico

### Teorema de Takens
El proyecto se basa en el teorema de reconstrucción del espacio de fases, que establece que una serie temporal escalar `s(t)` puede ser embebida en un espacio de dimensión superior que preserva las propiedades dinámicas del sistema original:

**X**ᵢ = [s(i), s(i+τ), s(i+2τ), ..., s(i+(d-1)τ)]

Donde:
- **τ** = retardo temporal (calculado por autocorrelación)
- **d** = dimensión de embebimiento (calculado por False Nearest Neighbors)

## 📁 Estructura del Proyecto

```
ModelosIA/
├── cloro/                           # Modelo de Concentración de Cloro
│   ├── generar_datos_cloro.py      # Genera serie temporal y guarda CSV
│   ├── modelo_cloro_lstm.py         # Modelo LSTM principal
│   ├── ejecutar_pipeline.py         # Script para ejecutar todo el pipeline
│   ├── comparar_modelos.py          # Comparación modelo físico vs LSTM
│   ├── README.md                    # Documentación del modelo de cloro
│   ├── datos_cloro.csv              # Serie temporal generada
│   ├── modelo_lstm_cloro.h5         # Modelo entrenado
│   ├── datos_cloro_visualizacion.png
│   ├── lstm_cloro_resultados.png
│   └── comparacion_modelo_fisico_vs_lstm.png
├── documentos/
│   └── guia_tecnica_reglas.md       # Guía de implementación
├── pyproject.toml                   # Configuración UV/Python
├── requirements.txt                 # Dependencias (pip)
├── INSTALACION_UV.md                # Guía de instalación con UV
└── README.md                        # Este archivo
```

## 🚀 Instalación

### Requisitos
- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Scikit-learn, Matplotlib, SciPy

### Opción 1: Instalar con UV (Recomendado - ⚡ Más rápido)

```bash
# Instalar UV si no lo tienes
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Instalar dependencias
uv venv
uv pip install -e .
```

Ver [INSTALACION_UV.md](INSTALACION_UV.md) para más detalles.

### Opción 2: Instalar con pip

```bash
pip install -r requirements.txt
```

## 💻 Uso

### Opción 1: Con UV (Recomendado)

```bash
# Navegar a la carpeta del modelo
cd cloro

# Pipeline completo
uv run --no-project ejecutar_pipeline.py

# O paso a paso
uv run --no-project generar_datos_cloro.py
uv run --no-project modelo_cloro_lstm.py
```

### Opción 2: Pipeline Completo (pip/entorno tradicional)

```bash
cd cloro
python ejecutar_pipeline.py
```

Este script ejecuta automáticamente:
1. Generación de datos
2. Entrenamiento del modelo LSTM
3. Evaluación y visualización

### Opción 3: Ejecución Paso a Paso

```bash
cd cloro

# Paso 1: Generar datos
python generar_datos_cloro.py

# Paso 2: Entrenar modelo
python modelo_cloro_lstm.py
```

## 🏗️ Arquitectura del Modelo

### Modelo LSTM (7 capas)

```
Capa 1: Dense(64, activation='relu')        # Proyección inicial
Capa 2: LSTM(64, return_sequences=True)      # Primera LSTM
Capa 3: Dropout(0.2)                         # Regularización
Capa 4: LSTM(64)                             # Segunda LSTM
Capa 5: Dropout(0.2)                         # Regularización
Capa 6: Dense(32, activation='relu')         # Capa intermedia
Capa 7: Dense(1)                             # Salida escalar
```

### Hiperparámetros

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Hidden Units | 64 | Balance entre capacidad y complejidad |
| Dropout Rate | 0.2 | Previene overfitting |
| Learning Rate | 0.001 | Convergencia estable |
| Batch Size | 16 | Apropiado para ~800 muestras |
| Épocas Máximas | 200 | Suficiente para convergencia |
| Early Stopping Patience | 15 | Restaura mejores pesos |

## 📊 Proceso de Modelado

### 1. Generación de Datos
- Sistema: EDO de concentración de cloro en tanque
- Parámetros físicos: k=0.15, C_entrada=2.0 mg/L
- Muestras: 1000 puntos temporales
- Ruido: Gaussiano (σ=0.02) para simular mediciones reales

### 2. Cálculo de Parámetros de Embebimiento

**τ (Retardo Temporal):**
- Método: Autocorrelación
- Criterio: Primer cruce por cero
- Rango esperado: [1, 10]

**d (Dimensión de Embebimiento):**
- Método: False Nearest Neighbors (FNN)
- Criterio: FNN < 5%
- Rango esperado: [4, 8]

### 3. División de Datos (SIN SHUFFLING)

```
Total: 100%
├── Train:      68%  (para aprendizaje)
├── Validation: 12%  (para early stopping)
└── Test:       20%  (para evaluación final)
```

### 4. Entrenamiento

- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE (Mean Squared Error)
- **Metric:** MAE (Mean Absolute Error)
- **Callback:** EarlyStopping (monitor='val_loss', patience=15)

### 5. Evaluación

**Métricas Cuantitativas:**
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- Expresadas como porcentaje del rango

**Criterios de Éxito:**
- ✓ RMSE < 5% del rango
- ✓ MAE < 3% del rango
- ✓ Predicción sigue observado visualmente

## 📈 Resultados Esperados

### Visualizaciones Generadas

1. **Serie Temporal Completa** (`datos_cloro_visualizacion.png`)
   - Serie temporal original
   - Distribución de valores

2. **Resultados del Modelo** (`lstm_cloro_resultados.png`)
   - Predicción vs Observado (Test Set)
   - Curvas de aprendizaje (Train/Val Loss)
   - Error de predicción
   - Distribución del error

### Métricas Típicas

Para el sistema de cloro (comportamiento suave y predecible):

```
RMSE: ~2-4% del rango
MAE:  ~1-3% del rango
Estado: ✓ APROBADO
```

## ✅ Reglas Cumplidas

### Reglas Generales
- **G1:** Separación caja negra (ecuaciones solo en generación, NO en entrenamiento)
- **G2:** Normalización MinMaxScaler a [0, 1]
- **G3:** Sin shuffling en datos temporales
- **G4:** Documentación completa de hiperparámetros

### Reglas de Embebimiento (Takens)
- **T1:** Cálculo de τ por autocorrelación
- **T2:** Cálculo de d por FNN
- **T3:** Construcción de matriz embebida

### Reglas LSTM
- **L1:** Arquitectura de 7 capas especificada
- **L2:** Compilación con Adam, MSE, MAE
- **L3:** Split 68/12/20 sin shuffling
- **L4:** Entrenamiento con early stopping
- **L5:** Evaluación en test set

### Reglas de Validación
- **V1:** Métricas numéricas (RMSE, MAE)
- **V2:** Validación visual con gráficas

## 🔍 Interpretación de Resultados

### ¿Qué significan las métricas?

- **RMSE < 5%:** El modelo predice con alta precisión
- **Predicción sigue observado:** La dinámica fue capturada correctamente
- **Convergencia sin overfitting:** Early stopping funcionó correctamente

### ¿Por qué funciona sin ecuaciones?

El **Teorema de Takens** garantiza que:
1. La serie temporal contiene toda la información del sistema
2. El embebimiento reconstruye el espacio de fases
3. La LSTM aprende las transiciones de estado

## 🧪 Validación Científica

El modelo es válido si:
1. ✓ RMSE < 5% del rango
2. ✓ Línea de predicción sigue observado visualmente
3. ✓ No hay divergencia en los primeros 5 pasos
4. ✓ Convergencia suave (sin overfitting agudo)

## 📚 Referencias

- Takens, F. (1981). "Detecting strange attractors in turbulence"
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- Kennel, Brown & Abarbanel (1992). "Determining embedding dimension"

## 👤 Autor

Proyecto desarrollado siguiendo la guía técnica para implementación de modelos IA en sistemas dinámicos.

## 📄 Licencia

Proyecto académico - Universidad

---

## 🆘 Solución de Problemas

### Error: "datos_cloro.csv no encontrado"
**Solución:** Ejecutar primero `python generar_datos_cloro.py`

### Advertencias de TensorFlow
**Solución:** Normal, se pueden ignorar

### Convergencia lenta
**Solución:** Verificar que τ y d sean razonables (τ < 10, d < 10)

### RMSE > 5%
**Causas posibles:**
- Datos insuficientes
- Hiperparámetros no optimizados
- Early stopping muy agresivo

---

**¡El modelo está listo para usar!** 🎉
