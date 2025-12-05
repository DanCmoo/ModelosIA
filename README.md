# Modelos de IA para Sistemas Dinámicos

## 📋 Descripción

Este proyecto implementa modelos de inteligencia artificial para predicción de sistemas dinámicos usando el **Teorema de Takens** para reconstrucción del espacio de fases. Incluye dos sistemas con diferentes características:

1. **Cloro (Sistema Predecible)** - LSTM
2. **Hindmarsh-Rose (Sistema Caótico)** - Echo State Network (ESN)

El proyecto sigue estrictamente las reglas y especificaciones de la guía técnica, demostrando que la IA puede aprender dinámicas **sin conocer las ecuaciones subyacentes** (enfoque de caja negra).

## 🎯 Objetivos

- Predecir comportamientos de sistemas dinámicos usando solo datos observados
- Comparar arquitecturas apropiadas para sistemas predecibles vs caóticos
- Demostrar la validez del Teorema de Takens en modelado de caja negra
- Cumplir con todas las reglas de implementación (G1-G4, T1-T3, L1-L5/E1-E7, V1-V5)

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
├── cloro/                           # Modelo de Concentración de Cloro (LSTM)
│   ├── generar_datos_cloro.py      # Genera serie temporal y guarda CSV
│   ├── modelo_cloro_lstm.py         # Modelo LSTM principal
│   ├── ejecutar_pipeline.py         # Script para ejecutar todo el pipeline
│   ├── comparar_modelos.py          # Comparación modelo físico vs LSTM
│   ├── README.md                    # Documentación del modelo de cloro
│   └── [archivos generados...]
├── hindmarsh_rose/                  # Modelo de Hindmarsh-Rose (ESN)
│   ├── generar_datos_hindmarsh_rose.py  # Genera serie caótica y guarda CSV
│   ├── modelo_hindmarsh_rose_esn.py     # Modelo ESN principal
│   ├── ejecutar_pipeline.py         # Script para ejecutar todo el pipeline
│   ├── README.md                    # Documentación del modelo H-R
│   └── [archivos generados...]
├── documentos/
│   └── guia_tecnica_reglas.md       # Guía de implementación completa
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

### Modelo de Cloro (LSTM)

```bash
# Navegar a la carpeta del modelo
cd cloro

# Con UV (Recomendado)
uv run --no-project ejecutar_pipeline.py

# O paso a paso
uv run --no-project generar_datos_cloro.py
uv run --no-project modelo_cloro_lstm.py
uv run --no-project comparar_modelos.py  # Comparación con modelo físico
```

### Modelo de Hindmarsh-Rose (ESN)

```bash
# Navegar a la carpeta del modelo
cd hindmarsh_rose

# Con UV (Recomendado)
uv run --no-project ejecutar_pipeline.py

# O paso a paso
uv run --no-project generar_datos_hindmarsh_rose.py
uv run --no-project modelo_hindmarsh_rose_esn.py
```

## 🏗️ Comparación de Arquitecturas

### Cloro vs Hindmarsh-Rose

| Aspecto | Cloro (LSTM) | Hindmarsh-Rose (ESN) |
|---------|--------------|---------------------|
| **Comportamiento** | Suave, predecible | Caótico, impredecible |
| **Arquitectura** | LSTM (6 capas) | ESN (Reservoir Computing) |
| **Parámetros entrenables** | ~52,000 | ~300 (solo W_out) |
| **Entrenamiento** | Backpropagation iterativo | Ridge Regression (solución cerrada) |
| **Tiempo de entrenamiento** | Minutos (~38 épocas) | Segundos |
| **Métrica clave** | RMSE < 5% | Horizonte Lyapunov > 5 pasos |
| **Predicción largo plazo** | Precisa (many-step) | Limitada (efecto mariposa) |
| **Reservoir size** | N/A | 300 neuronas |
| **Spectral radius** | N/A | 0.9 (edge of chaos) |

### Modelo LSTM para Cloro

```
Arquitectura: 6 capas
├─ Reshape(input_dim, 1)
├─ LSTM(64, return_sequences=True)
├─ Dropout(0.2)
├─ LSTM(64)
├─ Dropout(0.2)
├─ Dense(32, activation='relu')
└─ Dense(1)

Parámetros: ~52,000
Entrenamiento: Adam optimizer + Early Stopping
```

### Modelo ESN para Hindmarsh-Rose

```
Arquitectura: Reservoir Computing
├─ W_in (input → reservoir): Fijo, aleatorio
├─ W_res (reservoir): Fijo, ρ(W)=0.9
└─ W_out (reservoir → output): Entrenado (Ridge)

Parámetros entrenables: 300
Entrenamiento: Solución cerrada (Ridge Regression)
```

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
