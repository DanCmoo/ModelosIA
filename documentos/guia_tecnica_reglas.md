# GUÍA TÉCNICA: IMPLEMENTACIÓN DE MODELOS IA PARA SISTEMAS DINÁMICOS
## Reglas Precisas, Justificación Teórica y Código Implementable

---

## SECCIÓN 1: FUNDAMENTOS TEÓRICOS Y DECISIONES ARQUITECTÓNICAS

### 1.1 Base Teórica: Teorema de Takens

**Enunciado formal:**
Sea un sistema dinámico determinista de dimensión D con atractor extraño. Si se observa una única variable escalar s(t), entonces existe una dimensión de embebimiento d > 2D tal que la serie temporal embebida:

$$\mathbf{X}_i = [s_i, s_{i+\tau}, s_{i+2\tau}, ..., s_{i+(d-1)\tau}]$$

**es difeomorfa al atractor original**, preservando todas sus propiedades topológicas y dinámicas.

**Implicación para tu proyecto:**
- NO necesitas conocer las ecuaciones diferenciales
- La IA aprende del comportamiento observable (datos) únicamente
- La reconstrucción del espacio de fases es matemáticamente garantizada

**Justificación de uso:**
Este teorema es el fundamento que valida que tus modelos IA pueden capturar dinámicas sin ecuaciones explícitas.

---

### 1.2 Selección de Herramientas: Matriz de Decisión

| Característica | Cloro (Predecible) | Hindmarsh-Rose (Caótico) | Decisión |
|---|---|---|---|
| **Comportamiento** | Suave, monotónico | Transiciones abruptas, caótico | Requiere arquitecturas diferentes |
| **Dependencia temporal** | Mediano plazo (10-50 pasos) | Corto-largo plazo variable | LSTM vs ESN |
| **Predicción punto-exacto** | Viable > 20 pasos | Imposible > 10 pasos (Lyapunov) | Métricas diferentes |
| **Memoria necesaria** | Pesos aprendibles suficientes | Necesita reservorio dinámico rico | ESN superior |
| **Tiempo entrenamiento** | Aceptable (minutos) | Crítico (seconds/minutos) | ESN más rápida |

**Decisión final:**
- **Cloro** → **LSTM** (Long Short-Term Memory)
- **Hindmarsh-Rose** → **ESN** (Echo State Network)

---

## SECCIÓN 2: REGLAS EXPLÍCITAS PARA IMPLEMENTACIÓN

### 2.1 Reglas Generales (Aplican a ambos modelos)

#### Regla G1: Separación Caja Negra
```
PROHIBIDO: Usar ecuaciones diferenciales en el código del entrenamiento IA
PERMITIDO: Generar datos una sola vez, guardarlos en CSV, luego usar solo CSV

Pseudocódigo válido:
├─ Paso 1: generar_datos_cloro() → guardar a CSV
├─ Paso 2: cargar desde CSV
├─ Paso 3: entrenar modelo IA (SIN acceso a ecuación)
└─ Resultado: Modelo que aprendió solo de datos

Pseudocódigo INVÁLIDO:
├─ Generar datos dentro de loop de entrenamiento
├─ Acceder a ecuaciones durante predicción
└─ Validar contra ecuación analítica
```

**Justificación:** La asignación explícitamente prohíbe usar ecuaciones. Este protocolo lo cumple.

---

#### Regla G2: Normalización Obligatoria
```
PROCEDIMIENTO:
1. Cargar serie temporal sin procesar: X_raw = [x₁, x₂, ..., xₙ]
2. Aplicar MinMaxScaler a rango [0, 1]:
   X_norm = (X_raw - min(X_raw)) / (max(X_raw) - min(X_raw))
3. GUARDAR scaler para desnormalizar predicciones
4. Usar X_norm para todas las operaciones

CÓDIGO:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X_raw.reshape(-1, 1)).flatten()
```

**Justificación:** 
- Redes neuronales convergen mejor en rango [0,1]
- Evita overflow/underflow numérico
- Permite comparar sistemas de diferentes escalas

---

#### Regla G3: Nunca Shufflear Datos Temporales
```
PROHIBIDO:
X_train, X_test = train_test_split(X, y, shuffle=True)

OBLIGATORIO:
X_train, X_test = train_test_split(X, y, shuffle=False)

RAZÓN: Las series temporales tienen correlación secuencial.
Shufflear rompe la causalidad: predices pasado con futuro.
```

**Justificación:** Violar esto invalida toda conclusión sobre predicción temporal.

---

#### Regla G4: Documentación de Hiperparámetros
```
CADA modelo debe documentar:
├─ τ (retardo temporal): ¿Cómo se calculó?
├─ d (dimensión embebimiento): ¿Criterio FNN?
├─ Tamaño dataset: N muestras
├─ Split temporal: X% train, Y% val, Z% test
├─ Arquitectura: Capas, unidades, activaciones
├─ Entrenamiento: Optimizer, learning rate, épocas, callbacks
└─ Validación: Métrica primaria, criterio éxito

Debe estar en comentarios del código.
```

**Justificación:** Reproducibilidad científica.

---

### 2.2 Reglas Específicas: Embebimiento Temporal (Takens)

#### Regla T1: Cálculo de τ (Retardo Temporal)
```
ALGORITMO:
1. Calcular autocorrelación de serie normalizada
2. Buscar PRIMER punto donde ACF(τ) < 0 (cruce por cero)
3. Si no encuentra en 50 muestras: usar default τ = max_lag/4

PSEUDOCÓDIGO:
def estimar_tau(serie, max_lag=50):
    acf = autocorrelation(serie)
    for lag in range(1, max_lag):
        if acf[lag] < 0:
            return lag
    return max_lag // 4

IMPLEMENTACIÓN FÍSICA:
- Autocorrelación se calcula como:
  ACF(k) = ∑(sᵢ - s̄)(sᵢ₊ₖ - s̄) / ∑(sᵢ - s̄)²
- Usar numpy.correlate o scipy.signal.correlate
```

**Justificación teórica:** 
El primer cruce por cero indica el punto donde valores separados por τ dejan de ser correlacionados linealmente. Esto maximiza la información nueva en cada dimensión del embebimiento.

**Rango esperado:** τ ∈ [1, 10] para series uniformemente muestreadas

---

#### Regla T2: Cálculo de d (Dimensión de Embebimiento)
```
ALGORITMO: False Nearest Neighbors (FNN)

PARA cada dimensión d = 1, 2, 3, ..., d_max:
    1. Embeber serie en dimensión d:
       X_d = [s_i, s_{i+τ}, s_{i+2τ}, ..., s_{i+(d-1)τ}]
    
    2. Embeber en dimensión d+1:
       X_{d+1} = [s_i, s_{i+τ}, ..., s_{i+dτ}]
    
    3. PARA cada punto i en X_d:
       a) Encontrar vecino más cercano j: NN = argmin ||X_d[i] - X_d[j]||
       b) Calcular distancia en dim d: dist_d = ||X_d[i] - X_d[NN]||
       c) Calcular distancia en dim d+1: dist_{d+1} = ||X_{d+1}[i] - X_{d+1}[NN]||
       d) SI dist_{d+1} / dist_d > threshold (típicamente 20):
           - Marcar como "vecino falso" (se separa al aumentar dim)
    
    4. Calcular: FNN_percentage = (falsos / total) × 100%
    
    5. SI FNN_percentage < 5%:
       - RETORNAR d (dimensión óptima encontrada)

d_optimo garantiza: todos los falsos vecinos han desaparecido
```

**Justificación teórica:**
En dimensión demasiado baja, puntos cercanos en el espacio proyectado pueden estar lejanos en la realidad. Al aumentar d, esos "falsos cercanos" se separan. Cuando FNN < 5%, hemos capturado suficiente estructura.

**Rango esperado:**
- Cloro: d ∈ [4, 8]
- Hindmarsh-Rose: d ∈ [6, 12]

---

#### Regla T3: Creación de Matriz Embebida
```
ENTRADA: serie_normalizada, τ, d

PROCEDIMIENTO:
N = len(serie)
X = []  # Matriz de entrada
y = []  # Vector de salida

PARA i = 0 hasta N - d*τ - 1:
    vector_entrada = [serie[i], 
                      serie[i+τ], 
                      serie[i+2τ],
                      ...,
                      serie[i+(d-1)τ]]
    valor_salida = serie[i + d*τ]
    
    X.append(vector_entrada)
    y.append(valor_salida)

X = reshape(X) → (N - d*τ, d)
y = reshape(y) → (N - d*τ,)

RETORNAR X, y

JUSTIFICACIÓN:
X representa el estado del sistema en el espacio embebido
y es el siguiente paso que la red debe predecir
Esto crea un problema de predicción one-step-ahead
```

---

### 2.3 Reglas Específicas: LSTM para Cloro

#### Regla L1: Arquitectura de Red
```
ESPECIFICACIÓN:
├─ Capa 1: Dense(units=hidden_units, activation='relu')
│          input_shape = (d,)
│          Justificación: Proyección no lineal inicial
│
├─ Capa 2: LSTM(units=hidden_units, return_sequences=True)
│          Justificación: Primera capa LSTM con memoria
│
├─ Capa 3: Dropout(rate=0.2)
│          Justificación: Regularización, previene overfitting
│
├─ Capa 4: LSTM(units=hidden_units)
│          Justificación: Segunda capa LSTM sin return_sequences
│
├─ Capa 5: Dropout(rate=0.2)
│          Justificación: Regularización adicional
│
├─ Capa 6: Dense(units=32, activation='relu')
│          Justificación: Capa densa intermedia
│
└─ Capa 7: Dense(units=1)
           Justificación: Salida escalar (predicción siguiente valor)

VALORES RECOMENDADOS:
hidden_units = 64
dropout_rate = 0.2
```

**Justificación arquitectónica:**
- 2 capas LSTM capturan dependencias multi-escala
- Dropout evita overfitting en datos relativamente pequeños
- Dense final para mapeo no lineal final

---

#### Regla L2: Compilación del Modelo
```
CONFIGURACIÓN OBLIGATORIA:

optimizer = Adam(learning_rate=0.001)
  Justificación: Adapta learning rate por parámetro, converge rápido

loss = 'mse'  (Mean Squared Error)
  Justificación: Castiga desviaciones grandes, apropiado para predicción continua
  Fórmula: MSE = (1/N) ∑(ŷᵢ - yᵢ)²

metrics = ['mae']  (Mean Absolute Error)
  Justificación: Métrica interpretable en unidades originales
  Fórmula: MAE = (1/N) ∑|ŷᵢ - yᵢ|

Código:
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
```

---

#### Regla L3: Datos de Entrenamiento
```
PROCEDIMIENTO DE SPLIT:

1. X, y = crear matriz embebida (Regla T3)
   Tamaño: N muestras

2. SPLIT 1: Separar train+val vs test
   X_trainval, X_test, y_trainval, y_test = train_test_split(
       X, y, test_size=0.20, shuffle=False)
   
   Justificación: 20% test es estándar
   Tamaño test ≈ 0.2N

3. SPLIT 2: Dentro de trainval, separar train vs val
   X_train, X_val, y_train, y_val = train_test_split(
       X_trainval, y_trainval, test_size=0.15, shuffle=False)
   
   Justificación: 15% de trainval = ~12% del total
   
   Tamaño final:
   ├─ Train: 68% del total (0.8 × 0.85)
   ├─ Validation: 12% del total (0.8 × 0.15)
   └─ Test: 20% del total

NUNCA SHUFFLEAR: shuffle=False en ambos splits
```

---

#### Regla L4: Entrenamiento
```
PARÁMETROS OBLIGATORIOS:

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,                    # Máximo permitido
    batch_size=16,                 # Balance memoria/convergencia
    verbose=0,                     # Sin output durante entrenamiento
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=15,           # Parar si 15 épocas sin mejora
            restore_best_weights=True
        )
    ]
)

JUSTIFICACIÓN:
- epochs=200: Suficiente para convergencia en Cloro
- batch_size=16: Recomendado para N ~800 muestras
- EarlyStopping: Evita overfitting, restaura mejor modelo
- val_loss: Monitorear en validation, no train
```

---

#### Regla L5: Evaluación en Test
```
PROCEDIMIENTO OBLIGATORIO:

1. Predicción:
   y_pred = model.predict(X_test)

2. Cálculo de RMSE:
   rmse = sqrt(mean((y_test - y_pred)²))

3. Cálculo de MAE:
   mae = mean(|y_test - y_pred|)

4. Normalización de métrica:
   rango = max(serie_normalizada) - min(serie_normalizada)
   rmse_porcentaje = (rmse / rango) × 100

5. CRITERIO DE ÉXITO:
   ✓ rmse_porcentaje < 5%
   ✓ Gráfica predicción sigue observado visualmente
   ✓ Sin divergencia en primeros 5 pasos

CÓDIGO:
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
rmse_pct = (rmse / (y_max - y_min)) * 100
```

---

### 2.4 Reglas Específicas: ESN para Hindmarsh-Rose

#### Regla E1: Inicialización del Reservorio
```
PARÁMETROS CRÍTICOS:

reservoir_size = 300
Justificación: 
- Suficientemente grande para capturar dinámicas caóticas (típicamente 50-500)
- 300 es balance entre capacidad y complejidad computacional

spectral_radius = 0.9
Justificación:
- Controla el "edge of chaos": transición entre orden y caos
- ρ < 1: Sistema contrae, tiende a atractor fijo (orden)
- ρ ≈ 1: Sistema en borde del caos (óptimo para computación reservorio)
- ρ > 1: Sistema diverge (caótico puro)
- Para Hindmarsh-Rose caótico: 0.85-0.95 es óptimo

PROCEDIMIENTO DE INICIALIZACIÓN:

1. Generar matriz aleatoria:
   W_raw = normal_random(reservoir_size, reservoir_size)
   W_raw ~ N(0, 1)

2. Calcular radio espectral actual:
   eigenvalues = eigvals(W_raw)
   ρ_actual = max(|eigenvalues|)

3. Escalar matriz:
   W_res = (spectral_radius / ρ_actual) × W_raw
   
   Resultado: max(|eigvals(W_res)|) = spectral_radius

CÓDIGO:
W_raw = np.random.randn(reservoir_size, reservoir_size)
eigenvalues = np.linalg.eigvals(W_raw)
W_res = spectral_radius * W_raw / np.max(np.abs(eigenvalues))
```

**Justificación teórica:**
El radio espectral controla la memoria del reservorio. Valores cercanos a 1 permiten que información antigua influence predicciones nuevas, crucial para capturar dinámicas caóticas.

---

#### Regla E2: Inicialización de Pesos de Entrada
```
PARÁMETRO:
input_scale = 1.0

PROCEDIMIENTO:
W_in = input_scale × normal_random(reservoir_size, 1)
W_in ~ N(0, input_scale)

Justificación:
- Matriz (reservoir_size, 1): mapea entrada escalar a reservorio
- Valores aleatorios pequeños (1.0) evitan saturación de tanh

RANGO TÍPICO:
- input_scale ∈ [0.5, 2.0]
- 1.0 es default robusto

Efecto:
- input_scale pequeña: entrada débil, reservorio menos influenciado
- input_scale grande: entrada fuerte, puede saturar tanh
```

---

#### Regla E3: Dinámica del Reservorio
```
ECUACIÓN DE ACTUALIZACIÓN:

h(t+1) = tanh(W_res × h(t) + W_in × x(t))

Donde:
├─ h(t): vector de estado (reservoir_size,)
├─ W_res: matriz de recurrencia (fija, no entrenada)
├─ W_in: pesos de entrada (fijos, no entrenados)
├─ x(t): entrada escalar en tiempo t
├─ tanh: activación no lineal

INTERPRETACIÓN:
1. W_res @ h(t): dinámica interna, mezcla de toda la historia
2. W_in @ x(t): nueva información de entrada
3. tanh: comprime a [-1, 1], introduce no linealidad
4. h(t+1) NUEVO: estado actualizado listo para siguiente paso

PSEUDOCÓDIGO:
def step_reservorio(h, x_new):
    h_nuevo = np.tanh(W_res @ h + W_in @ x_new)
    return h_nuevo

JUSTIFICACIÓN:
- tanh (vs. relu): Preserva dinámica suave, apropiado para caos
- W_res fija: Evita entrenamiento costoso, captura fenómenos complejos
- Recurrencia: Permite memoria de largo plazo
```

---

#### Regla E4: Generación de Estados (Training)
```
PROCEDIMIENTO:

ENTRADA: X_train (N_samples, d), matriz embebida

INICIALIZACIÓN:
h = zeros(reservoir_size)  # Estado inicial en cero

LOOP:
states = []
PARA cada fila i en X_train:
    x_t = X_train[i, :]        # Vector de entrada d-dimensional
    h = tanh(W_res @ h + W_in @ x_t)
    states.append(h)           # Guardar estado

RESULTADO:
states = (N_samples, reservoir_size)

JUSTIFICACIÓN:
- Propagamos entrada a través del reservorio
- Almacenamos todos los estados internos
- Usaremos estos estados para entrenar W_out

CÓDIGO PYTHON:
def generate_states(X_train, W_res, W_in, reservoir_size):
    n_samples = X_train.shape[0]
    states = np.zeros((n_samples, reservoir_size))
    h = np.zeros(reservoir_size)
    
    for i in range(n_samples):
        h = np.tanh(W_res @ h + W_in @ X_train[i].reshape(-1, 1))
        states[i] = h.flatten()
    
    return states
```

---

#### Regla E5: Entrenamiento de Pesos de Salida (Ridge Regression)
```
PROBLEMA:
Encontrar W_out tal que: H @ W_out ≈ y_train

Donde:
├─ H: matriz de estados (N_samples, reservoir_size)
├─ W_out: pesos a entrenar (reservoir_size, 1)
├─ y_train: valores objetivo (N_samples,)

SOLUCIÓN: Ridge Regression (regularizada)

W_out = (H^T H + λI)^{-1} H^T y_train

Donde:
├─ H^T H: matriz de Gram (reservoir_size, reservoir_size)
├─ λ: parámetro regularización (típicamente 1e-6)
├─ I: matriz identidad

JUSTIFICACIÓN:
- Ridge regression: Add penalty term λ||W_out||² para evitar overfitting
- Solución cerrada: No requiere gradient descent iterativo
- Rápido: O(d³) donde d = reservoir_size

CÓDIGO PYTHON:
H = generate_states(X_train, W_res, W_in, reservoir_size)

H_T_H = H.T @ H
H_T_y = H.T @ y_train

lambda_reg = 1e-6
I = np.eye(reservoir_size)

W_out = np.linalg.solve(
    H_T_H + lambda_reg * I,
    H_T_y
)

VALIDACIÓN:
print(f"W_out shape: {W_out.shape}")  # Debe ser (reservoir_size, 1)
```

---

#### Regla E6: Predicción One-Step-Ahead
```
PROPÓSITO:
Validar que ESN aprendió dinámicas en test set
Sin retroalimentación: usa valores observados reales

PROCEDIMIENTO:

ENTRADA: X_test (N_test, d), matriz embebida test

H_test = generate_states(X_test, W_res, W_in, reservoir_size)
y_pred = H_test @ W_out

RESULTADO:
y_pred: (N_test,) predicciones one-step

ERROR METRICS:
rmse = sqrt(mean((y_test - y_pred)²))
mae = mean(|y_test - y_pred|)

CRITERIO ÉXITO (Hindmarsh-Rose):
✓ rmse < 10% del rango

JUSTIFICACIÓN:
- One-step es posible incluso en caos porque no retroalimentamos
- Método estándar para validar aprendizaje base
```

---

#### Regla E7: Predicción Multi-Step (Retroalimentada)
```
PROPÓSITO:
Medir horizonte de Lyapunov: cuántos pasos predecimos coherentemente

PROCEDIMIENTO:

ENTRADA: x_init (d,) vector inicial embebido

predictions = []
h = zeros(reservoir_size)
x = x_init

PARA t = 1 hasta max_steps:
    h = tanh(W_res @ h + W_in @ x)
    y_next = W_out @ h
    predictions.append(y_next)
    x = array([y_next])  # Retroalimenta predicción
    
RESULTADO:
predictions: (max_steps,) predicciones futuras

CÁLCULO DEL HORIZONTE DE LYAPUNOV:

error_relativo = |predictions - y_true| / (max - min)
lyapunov_horizon = argmax{t : error_relativo[t] < 0.5}

Interpetación:
- Mientras error < 50%: predicción "útil"
- Cuando error > 50%: predicción completamente divergida
- Típico para HR: 5-15 pasos

CÓDIGO:
def predict_multistep(x_init, W_res, W_in, W_out, 
                      reservoir_size, steps=50):
    predictions = []
    h = np.zeros(reservoir_size)
    x = x_init.reshape(-1, 1)
    
    for _ in range(steps):
        h = np.tanh(W_res @ h + W_in @ x)
        y_next = W_out @ h
        predictions.append(y_next[0, 0])
        x = y_next
    
    return np.array(predictions)
```

---

## SECCIÓN 3: VALIDACIÓN Y MÉTRICAS

### 3.1 Validación para Cloro (Método Cuantitativo)

#### V1: Métricas Numéricas
```
RMSE (Root Mean Squared Error):
rmse = sqrt((1/N) ∑(ŷᵢ - yᵢ)²)

Criterio: RMSE < 5% del rango
Interpretación: Pequeños errores sistemáticamente

MAE (Mean Absolute Error):
mae = (1/N) ∑|ŷᵢ - yᵢ|

Criterio: MAE < 3% del rango
Interpretación: Error promedio en unidades originales

MAPE (Mean Absolute Percentage Error):
mape = (100/N) ∑|((ŷᵢ - yᵢ) / yᵢ)|

Criterio: MAPE < 10%
Interpretación: Error porcentual promedio

CÓDIGO:
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

rango = np.max(y_test) - np.min(y_test)
rmse_pct = (rmse / rango) * 100
mae_pct = (mae / rango) * 100

print(f"RMSE: {rmse:.6f} ({rmse_pct:.2f}%)")
print(f"MAE:  {mae:.6f} ({mae_pct:.2f}%)")
```

#### V2: Validación Visual
```
GRÁFICA REQUERIDA:

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(len(y_test)), y_test, 'b-', 
         label='Observado', linewidth=2)
plt.plot(range(len(y_pred)), y_pred, 'r--', 
         label='Predicho LSTM', linewidth=2, alpha=0.7)
plt.xlabel('Tiempo (pasos)')
plt.ylabel('Concentración (normalizado)')
plt.title('Test Set: Predicción vs Observado')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('Convergencia del Entrenamiento')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_validacion.png', dpi=150)

CRITERIOS VISUALES:
- Línea roja sigue línea azul ✓
- Sin divergencia notable
- Convergencia suave (sin overfitting agudo)
```

---

### 3.2 Validación para Hindmarsh-Rose (Método Cualitativo)

#### V3: Horizonte de Lyapunov
```
DEFINICIÓN:
Número máximo de pasos donde el error relativo permanece < 50%

CÁLCULO:
error_rel[t] = |ŷ[t] - y[t]| / (max(y) - min(y))
lyapunov_horizon = max{t : error_rel[t] < 0.5}

CRITERIO:
✓ lyapunov_horizon > 5 pasos (mínimo)
✓ Idealmente 8-15 pasos

INTERPRETACIÓN:
- ESN mantiene información coherente ~10 pasos
- Después: efecto mariposa domina, predicción diverge
- Esto es ESPERADO en sistemas caóticos

CÓDIGO:
error_rel = np.abs(predictions - y_test[:len(predictions)]) / \
            (np.max(y_test) - np.min(y_test))
threshold_indices = np.where(error_rel > 0.5)[0]
if len(threshold_indices) > 0:
    lyapunov_horizon = threshold_indices[0]
else:
    lyapunov_horizon = len(predictions)
```

#### V4: Propiedades Estadísticas
```
COMPARACIÓN: Serie Observada vs Predicha (multi-step)

1. MEDIA:
   mean_obs = mean(y_observado)
   mean_pred = mean(y_predicho)
   error_media = |mean_obs - mean_pred| / |mean_obs| × 100
   
   Criterio: error_media < 10%

2. DESVIACIÓN ESTÁNDAR:
   std_obs = std(y_observado)
   std_pred = std(y_predicho)
   error_std = |std_obs - std_pred| / std_obs × 100
   
   Criterio: error_std < 15%

3. AUTOCORRELACIÓN:
   acf_obs = autocorrelation(y_observado, max_lag=20)
   acf_pred = autocorrelation(y_predicho, max_lag=20)
   acf_error = mean((acf_obs - acf_pred)²)
   
   Criterio: acf_error < 0.05

JUSTIFICACIÓN:
- Si ESN capturó la dinámica caótica correctamente,
  debe reproducir estas propiedades estadísticas
- La predicción exacta es imposible, pero estadísticas deben coincidir

CÓDIGO:
mean_obs = np.mean(y_observado)
mean_pred = np.mean(y_predicho)
error_media_pct = abs(mean_obs - mean_pred) / abs(mean_obs) * 100

std_obs = np.std(y_observado)
std_pred = np.std(y_predicho)
error_std_pct = abs(std_obs - std_pred) / std_obs * 100

print(f"Error media: {error_media_pct:.2f}%")
print(f"Error std:   {error_std_pct:.2f}%")
```

#### V5: Atractor en Espacio 3D
```
RECONSTRUCCIÓN DEL ATRACTOR:

OBSERVADO:
X₁ = [y[0], y[0+τ], y[0+2τ]]
X₂ = [y[1], y[1+τ], y[1+2τ]]
...
Xₙ = [y[n], y[n+τ], y[n+2τ]]

PREDICHO (multi-step):
Ŷ₁ = [ŷ[0], ŷ[0+τ], ŷ[0+2τ]]
...
Ŷₙ = [ŷ[n], ŷ[n+τ], ŷ[n+2τ]]

VISUALIZACIÓN 3D:
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(X[:, 0], X[:, 1], X[:, 2], 'b-', alpha=0.6, linewidth=0.5)
ax1.set_xlabel('y(t)')
ax1.set_ylabel('y(t+τ)')
ax1.set_zlabel('y(t+2τ)')
ax1.set_title('Atractor Observado')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(Ŷ[:, 0], Ŷ[:, 1], Ŷ[:, 2], 'r-', alpha=0.6, linewidth=0.5)
ax2.set_xlabel('ŷ(t)')
ax2.set_ylabel('ŷ(t+τ)')
ax2.set_zlabel('ŷ(t+2τ)')
ax2.set_title('Atractor Predicho (ESN)')

plt.tight_layout()
plt.savefig('atractor_comparacion_3d.png', dpi=150)

CRITERIO VISUAL:
✓ Ambos atractores tienen forma similar
✓ Estructura topológica preservada
✓ NO es necesario que coincidan exactamente

JUSTIFICACIÓN:
El Teorema de Takens garantiza que la topología del atractor
se preserva bajo embebimiento. Si ESN capturó dinámicas,
la forma 3D debe ser similar.
```

---

## SECCIÓN 4: EXPERIMENTO FALLIDO CON SVM (Justificación)

### 4.1 Por qué SVM debe fallar

```
ARQUITECTURA SVM:
ŷ = f(X) = ∑ αᵢ K(X, Xᵢ) + b

Donde K es kernel (rbf típicamente)

LIMITACIÓN FUNDAMENTAL:
- f es FUNCIÓN ESTÁTICA: mismo X → siempre mismo ŷ
- NO hay memoria de estados previos
- NO hay dinámica interna

EN HINDMARSH-ROSE CAÓTICO:

Paso 1: Predecir con X_test[0], obtener ŷ[1] con error ε₁
Paso 2: Retroalimentar ŷ[1], pero SVM NO sabe que había error
        El nuevo X_test[1]_modificado da predicción ŷ[2]
        Error acumulado: ε₂ ≈ λ × ε₁ (λ = exponente Lyapunov)
Paso 3: Error crece exponencialmente: εₜ ≈ e^(λt) × ε₁

RESULTADO: Divergencia rápida (típicamente 2-5 pasos)

COMPARACIÓN SVM vs ESN:

│ Aspecto │ SVM │ ESN │
│────────────────────────────────────────│
│ Memoria │ 0 (estático) │ ∞ (reservorio) │
│ Error multi-step │ Exponencial │ Lineal/contenido │
│ Horizonte Lyapunov │ ~2 pasos │ ~10 pasos │
│ Aplicable caos │ NO │ SÍ │
```

### 4.2 Protocolo de Experimento Fallido

```
PASO 1: Entrenar SVM idéntico a ESN
svm = SVR(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_hr, y_train_hr)

PASO 2: Predicción one-step
y_pred_svm_1step = svm.predict(X_test_hr)
rmse_svm = sqrt(mean((y_test_hr - y_pred_svm_1step)²))

Resultado esperado: rmse_svm ~ rmse_esn (ambos entrenan igual)

PASO 3: Predicción multi-step
predictions_svm = []
x = X_test_hr[0]

for i in range(50):
    y_next = svm.predict(x.reshape(1, -1))[0]
    predictions_svm.append(y_next)
    x = np.roll(x, -1)  # Desplaza ventana
    x[-1] = y_next      # Añade predicción

PASO 4: Analizar divergencia
error_svm = |predictions_svm - y_test[:50]|
rel_error_svm = error_svm / (max(y) - min(y))

para paso en [5, 10, 20, 50]:
    print(f"Error relativo SVM paso {paso}: {rel_error_svm[paso]*100:.1f}%")

Resultado esperado:
- Paso 5: ~20% error
- Paso 10: ~50% error (cruzó umbral)
- Paso 20: ~99% error (divergió)
- Paso 50: >100% error (completamente incoherente)

VISUALIZACIÓN:
plt.figure(figsize=(12, 5))
plt.plot(range(50), y_test_hr[:50], 'b-', label='Observado', linewidth=2)
plt.plot(range(50), predictions_esn, 'g-', label='ESN', alpha=0.7)
plt.plot(range(50), predictions_svm, 'r--', label='SVM', alpha=0.7)
plt.xlabel('Pasos de predicción')
plt.ylabel('Voltaje')
plt.title('Comparación Multi-step: ESN vs SVM en Hindmarsh-Rose')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('svm_vs_esn_falla.png', dpi=150)

CONCLUSIÓN EN REPORTE:
"SVM, siendo un modelo estático sin mecanismo de memoria,
es fundamentalmente inadecuado para predicción en sistemas
caóticos. Su error crece exponencialmente debido al efecto
mariposa (sensibilidad a condiciones iniciales). ESN, con su
reservorio dinámico, mantiene coherencia predictiva hasta
~10 pasos, demostrando su superioridad para dinámicas caóticas."
```

---

## SECCIÓN 5: ESTRUCTURA DE CÓDIGO FINAL

```python
"""
PROYECTO: MODELADO CAJA NEGRA DE SISTEMAS DINÁMICOS
Usando LSTM (Cloro) y ESN (Hindmarsh-Rose)

Reglas de implementación:
1. NO usar ecuaciones en entrenamiento IA
2. Separación estricta: generación de datos vs aprendizaje
3. Toda decisión debe estar justificada teóricamente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============== BLOQUE 1: DATOS (CAJA NEGRA VERIFICABLE) ==============

def generar_datos_caja_negra():
    """
    Genera series temporales UNA SOLA VEZ
    Resultados se guardan en CSV (accesibles solo a IA)
    """
    # [Implementar según especificación en Sección 1]
    pass

# ============== BLOQUE 2: EMBEBIMIENTO TAKENS ==============

class TakensEmbedding:
    """
    Implementa reconstrucción de espacio de fases
    Reglas: T1, T2, T3
    """
    @staticmethod
    def estimar_tau(serie):
        # Regla T1: Primer cruce por cero autocorrelación
        pass
    
    @staticmethod
    def false_nearest_neighbors(serie, tau):
        # Regla T2: FNN < 5%
        pass
    
    @staticmethod
    def embebe(serie, tau, d):
        # Regla T3: Crear matriz (N, d)
        pass

# ============== BLOQUE 3: LSTM PARA CLORO ==============

class LSTMCloro:
    """
    Modelo LSTM para concentración de cloro
    Reglas: L1-L5
    """
    def __init__(self, embedding_dim):
        self.model = self._construir_arquitectura(embedding_dim)
    
    def _construir_arquitectura(self, d):
        # Regla L1: Arquitectura específica
        pass
    
    def compilar(self):
        # Regla L2: Compilación
        pass
    
    def entrenar(self, X_train, X_val, y_train, y_val):
        # Regla L4: Entrenamiento con early stopping
        pass
    
    def evaluar(self, X_test, y_test):
        # Regla L5: Evaluación, calcular RMSE
        pass

# ============== BLOQUE 4: ESN PARA HINDMARSH-ROSE ==============

class EchoStateNetwork:
    """
    Red de Estado Eco para Hindmarsh-Rose
    Reglas: E1-E7
    """
    def __init__(self, reservoir_size=300, spectral_radius=0.9):
        # Regla E1: Inicializar reservorio
        self.W_res = self._init_reservoir(reservoir_size, spectral_radius)
        # Regla E2: Inicializar entrada
        self.W_in = self._init_input(reservoir_size)
        self.W_out = None
    
    def _init_reservoir(self, size, rho):
        # E1: Escalar por radio espectral
        pass
    
    def _init_input(self, size):
        # E2: Pesos entrada aleatorios
        pass
    
    def fit(self, X_train, y_train):
        # E3, E4, E5: Propagar + Ridge Regression
        pass
    
    def predict_onestep(self, X_test):
        # E6: Predicción sin retroalimentación
        pass
    
    def predict_multistep(self, x_init, steps):
        # E7: Predicción retroalimentada
        pass

# ============== BLOQUE 5: VALIDACIÓN ==============

class Validacion:
    """
    Métricas y visualización
    Reglas: V1-V5
    """
    @staticmethod
    def validar_cloro(y_test, y_pred):
        # V1, V2: RMSE, MAE, gráficas
        pass
    
    @staticmethod
    def validar_hindmarsh_rose_multistep(y_test, y_pred_multistep, 
                                         tau, d):
        # V3, V4, V5: Lyapunov, estadísticas, atractor 3D
        pass

# ============== BLOQUE 6: EXPERIMENTO FALLIDO ==============

def experimento_svm_fallido(X_train_hr, X_test_hr, y_train_hr, y_test_hr):
    """
    Demuestra por qué SVM falla en sistemas caóticos
    Regla: Sección 4
    """
    pass

if __name__ == "__main__":
    print("="*70)
    print("EJECUTANDO PROYECTO: MODELADO CAJA NEGRA")
    print("="*70)
    
    # 1. Generar datos y guardar CSV
    generar_datos_caja_negra()
    
    # 2. Embeber series temporales
    # 3. Entrenar LSTM (Cloro)
    # 4. Entrenar ESN (Hindmarsh-Rose)
    # 5. Validar ambos
    # 6. Comparar con SVM fallido
    
    print("\n✓ PROYECTO COMPLETADO")
```

---

## SECCIÓN 6: CHECKLIST FINAL DE IMPLEMENTACIÓN

```
ANTES DE ENTREGAR:

[ ] Datos generados una sola vez, guardados en CSV
[ ] Embebimiento de Takens implementado (τ, d calculados)
[ ] LSTM entrenado en Cloro con RMSE < 5%
[ ] ESN entrenado en Hindmarsh-Rose
[ ] Horizonte Lyapunov calculado (>5 pasos)
[ ] Propiedades estadísticas comparadas
[ ] Atractor 3D visualizado y comparado
[ ] SVM demostrado como fallido (divergencia exponencial)
[ ] Todas las líneas de código justificadas en comentarios
[ ] Gráficas guardadas: lstm_cloro.png, atractor_3d.png, svm_falla.png
[ ] Cada hiperparámetro documentado (τ, d, reservoir_size, etc.)
[ ] Reporte teórico escrito (este documento como referencia)

CRITERIOS DE ÉXITO:

✓ LSTM: RMSE_cloro < 5% del rango
✓ ESN: Lyapunov_horizon > 5 pasos
✓ ESN: Estadísticas (media/std) dentro ±15% de observado
✓ SVM: Diverge notoriamente antes que ESN
✓ Código reproducible: Ejecutar script genera todos resultados
✓ Caja negra verificable: Sin acceso a ecuaciones en IA
```

---

**Documento creado por:** Sistema de Asesoramiento Técnico
**Versión:** 1.0
**Fecha:** Diciembre 2025
**Destinatario:** Proyecto Final - Cibernetica y IA
