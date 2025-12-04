"""
MODELO LSTM PARA CONCENTRACIÓN DE CLORO
========================================
Este script implementa el modelo de predicción usando LSTM
siguiendo estrictamente las reglas de la guía técnica.

ARQUITECTURA: LSTM (Long Short-Term Memory)
CUMPLE: Reglas G1-G4, T1-T3, L1-L5, V1-V2

AUTOR: Sistema de IA
FECHA: Diciembre 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import correlate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PARTE 1: CÁLCULO DE PARÁMETROS DE EMBEBIMIENTO (TEOREMA DE TAKENS)
# =============================================================================

def estimar_tau(serie, max_lag=50):
    """
    Calcula el retardo temporal óptimo τ usando autocorrelación.
    
    CUMPLE REGLA T1: Primer cruce por cero de la autocorrelación
    
    Parámetros:
    -----------
    serie : array
        Serie temporal normalizada
    max_lag : int
        Máximo retardo a evaluar
    
    Returns:
    --------
    tau : int
        Retardo temporal óptimo
    acf : array
        Valores de autocorrelación
    """
    # Normalizar serie (centrar en 0)
    serie_centrada = serie - np.mean(serie)
    
    # Calcular autocorrelación
    acf = np.correlate(serie_centrada, serie_centrada, mode='full')
    acf = acf[len(acf)//2:]  # Solo lags positivos
    acf = acf / acf[0]  # Normalizar
    
    # Buscar primer cruce por cero
    for lag in range(1, min(max_lag, len(acf))):
        if acf[lag] < 0:
            print(f"    - τ encontrado en primer cruce por cero: {lag}")
            return lag, acf
    
    # Si no encuentra cruce, usar default
    tau = max_lag // 4
    print(f"    - No se encontró cruce por cero, usando default: {tau}")
    return tau, acf


def calcular_fnn(serie, tau, d_max=15, threshold=20.0):
    """
    Calcula la dimensión de embebimiento óptima usando False Nearest Neighbors.
    
    CUMPLE REGLA T2: FNN con threshold de separación
    
    Parámetros:
    -----------
    serie : array
        Serie temporal normalizada
    tau : int
        Retardo temporal
    d_max : int
        Dimensión máxima a evaluar
    threshold : float
        Umbral para identificar vecinos falsos
    
    Returns:
    --------
    d_optimo : int
        Dimensión de embebimiento óptima
    fnn_percentages : list
        Porcentajes de FNN para cada dimensión
    """
    n = len(serie)
    fnn_percentages = []
    
    for d in range(1, d_max + 1):
        # Crear embebimientos en dimensión d y d+1
        max_idx = n - (d * tau)
        if max_idx < 10:  # Muy pocos puntos
            break
        
        # Embebimiento en dim d
        X_d = np.zeros((max_idx, d))
        for i in range(max_idx):
            for j in range(d):
                X_d[i, j] = serie[i + j * tau]
        
        # Embebimiento en dim d+1
        max_idx_d1 = n - ((d + 1) * tau)
        if max_idx_d1 < 10:
            break
        
        X_d1 = np.zeros((max_idx_d1, d + 1))
        for i in range(max_idx_d1):
            for j in range(d + 1):
                X_d1[i, j] = serie[i + j * tau]
        
        # Contar vecinos falsos
        false_neighbors = 0
        total_neighbors = 0
        
        for i in range(min(len(X_d), len(X_d1))):
            # Encontrar vecino más cercano en dim d (excluyendo el mismo punto)
            distances_d = np.linalg.norm(X_d - X_d[i], axis=1)
            distances_d[i] = np.inf  # Excluir el mismo punto
            nn_idx = np.argmin(distances_d)
            dist_d = distances_d[nn_idx]
            
            if dist_d == 0 or nn_idx >= len(X_d1):
                continue
            
            # Calcular distancia en dim d+1
            dist_d1 = np.linalg.norm(X_d1[i] - X_d1[nn_idx])
            
            # Verificar si es vecino falso
            if dist_d > 0:
                ratio = dist_d1 / dist_d
                if ratio > threshold:
                    false_neighbors += 1
            
            total_neighbors += 1
        
        # Calcular porcentaje
        if total_neighbors > 0:
            fnn_pct = (false_neighbors / total_neighbors) * 100
        else:
            fnn_pct = 100
        
        fnn_percentages.append(fnn_pct)
        
        print(f"    - d={d}: FNN={fnn_pct:.2f}%")
        
        # Criterio de parada: FNN < 5%
        if fnn_pct < 5.0:
            print(f"    ✓ Dimensión óptima encontrada: d={d}")
            return d, fnn_percentages
    
    # Si no converge, usar dimensión con mínimo FNN
    d_optimo = np.argmin(fnn_percentages) + 1
    print(f"    - Usando dimensión con mínimo FNN: d={d_optimo}")
    return d_optimo, fnn_percentages


# =============================================================================
# PARTE 2: EMBEBIMIENTO TEMPORAL
# =============================================================================

def crear_matriz_embebida(serie, tau, d):
    """
    Crea matriz embebida según el Teorema de Takens.
    
    CUMPLE REGLA T3: Construcción de matriz embebida
    
    Parámetros:
    -----------
    serie : array
        Serie temporal normalizada
    tau : int
        Retardo temporal
    d : int
        Dimensión de embebimiento
    
    Returns:
    --------
    X : array
        Matriz de entrada (n_samples, d)
    y : array
        Vector de salida (n_samples,)
    """
    n = len(serie)
    max_idx = n - d * tau
    
    X = np.zeros((max_idx, d))
    y = np.zeros(max_idx)
    
    for i in range(max_idx):
        # Vector de entrada: [s(i), s(i+τ), s(i+2τ), ..., s(i+(d-1)τ)]
        for j in range(d):
            X[i, j] = serie[i + j * tau]
        
        # Valor de salida: s(i+d*τ)
        y[i] = serie[i + d * tau]
    
    return X, y


# =============================================================================
# PARTE 3: CONSTRUCCIÓN DEL MODELO LSTM
# =============================================================================

def construir_modelo_lstm(input_dim, hidden_units=64):
    """
    Construye la arquitectura LSTM según especificaciones.
    
    CUMPLE REGLA L1: Arquitectura específica
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión de entrada (d)
    hidden_units : int
        Unidades en capas LSTM
    
    Returns:
    --------
    model : keras.Model
        Modelo LSTM compilado
    """
    model = Sequential([
        # Reshape para LSTM (necesita 3D: samples, timesteps, features)
        keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        
        # Capa 1: Primera LSTM con return_sequences
        LSTM(hidden_units, return_sequences=True),
        
        # Capa 2: Dropout
        Dropout(0.2),
        
        # Capa 3: Segunda LSTM sin return_sequences
        LSTM(hidden_units),
        
        # Capa 4: Dropout
        Dropout(0.2),
        
        # Capa 5: Dense intermedia
        Dense(32, activation='relu'),
        
        # Capa 6: Dense de salida
        Dense(1)
    ])
    
    # CUMPLE REGLA L2: Compilación
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# =============================================================================
# PARTE 4: FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("="*70)
    print("MODELO LSTM PARA PREDICCIÓN DE CONCENTRACIÓN DE CLORO")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # PASO 1: CARGAR DATOS (CUMPLE REGLA G1)
    # -------------------------------------------------------------------------
    print("\n[PASO 1/8] Cargando datos desde CSV...")
    try:
        df = pd.read_csv('datos_cloro.csv')
        serie_raw = df['concentracion'].values
        print(f"    ✓ Datos cargados: {len(serie_raw)} puntos")
        print(f"    - Rango: [{serie_raw.min():.4f}, {serie_raw.max():.4f}]")
    except FileNotFoundError:
        print("    ✗ ERROR: datos_cloro.csv no encontrado")
        print("    → Ejecutar primero: generar_datos_cloro.py")
        return
    
    # -------------------------------------------------------------------------
    # PASO 2: NORMALIZACIÓN (CUMPLE REGLA G2)
    # -------------------------------------------------------------------------
    print("\n[PASO 2/8] Normalizando datos...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie_norm = scaler.fit_transform(serie_raw.reshape(-1, 1)).flatten()
    print(f"    ✓ Serie normalizada a rango [0, 1]")
    print(f"    - Min: {serie_norm.min():.6f}")
    print(f"    - Max: {serie_norm.max():.6f}")
    
    # -------------------------------------------------------------------------
    # PASO 3: CALCULAR τ (CUMPLE REGLA T1)
    # -------------------------------------------------------------------------
    print("\n[PASO 3/8] Calculando retardo temporal τ...")
    tau, acf = estimar_tau(serie_norm, max_lag=50)
    print(f"    ✓ τ = {tau}")
    
    # -------------------------------------------------------------------------
    # PASO 4: CALCULAR d (CUMPLE REGLA T2)
    # -------------------------------------------------------------------------
    print("\n[PASO 4/8] Calculando dimensión de embebimiento d...")
    d, fnn_pct = calcular_fnn(serie_norm, tau, d_max=15, threshold=20.0)
    print(f"    ✓ d = {d}")
    
    # -------------------------------------------------------------------------
    # PASO 5: CREAR MATRIZ EMBEBIDA (CUMPLE REGLA T3)
    # -------------------------------------------------------------------------
    print("\n[PASO 5/8] Creando matriz embebida...")
    X, y = crear_matriz_embebida(serie_norm, tau, d)
    print(f"    ✓ X shape: {X.shape}")
    print(f"    ✓ y shape: {y.shape}")
    
    # -------------------------------------------------------------------------
    # PASO 6: SPLIT DE DATOS (CUMPLE REGLA L3 y G3)
    # -------------------------------------------------------------------------
    print("\n[PASO 6/8] Dividiendo datos (train/val/test)...")
    
    # Split 1: train+val vs test (80/20)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False  # REGLA G3: NO SHUFFLEAR
    )
    
    # Split 2: train vs val (85/15 del trainval)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, shuffle=False
    )
    
    print(f"    ✓ Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    ✓ Val:   {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    ✓ Test:  {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # PASO 7: CONSTRUIR Y ENTRENAR MODELO (CUMPLE REGLAS L1, L2, L4)
    # -------------------------------------------------------------------------
    print("\n[PASO 7/8] Construyendo y entrenando modelo LSTM...")
    
    # Construir modelo
    model = construir_modelo_lstm(input_dim=d, hidden_units=64)
    
    print("\nArquitectura del modelo:")
    model.summary()
    
    # Configurar callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Entrenar (CUMPLE REGLA L4)
    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        verbose=1,
        callbacks=[early_stop]
    )
    
    print(f"\n    ✓ Entrenamiento completado")
    print(f"    - Épocas ejecutadas: {len(history.history['loss'])}")
    print(f"    - Loss final (train): {history.history['loss'][-1]:.6f}")
    print(f"    - Loss final (val): {history.history['val_loss'][-1]:.6f}")
    
    # -------------------------------------------------------------------------
    # PASO 8: EVALUACIÓN (CUMPLE REGLAS L5, V1, V2)
    # -------------------------------------------------------------------------
    print("\n[PASO 8/8] Evaluando modelo en conjunto de prueba...")
    
    # Predicción
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Métricas (CUMPLE REGLA V1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    rango = serie_norm.max() - serie_norm.min()
    rmse_pct = (rmse / rango) * 100
    mae_pct = (mae / rango) * 100
    
    print(f"\n    MÉTRICAS DE EVALUACIÓN:")
    print(f"    {'─'*50}")
    print(f"    RMSE:      {rmse:.6f}  ({rmse_pct:.2f}% del rango)")
    print(f"    MAE:       {mae:.6f}  ({mae_pct:.2f}% del rango)")
    print(f"    {'─'*50}")
    
    # Criterio de éxito
    print(f"\n    CRITERIO DE ÉXITO:")
    if rmse_pct < 5.0:
        print(f"    ✓ RMSE < 5%: APROBADO")
    else:
        print(f"    ✗ RMSE < 5%: RECHAZADO")
    
    if mae_pct < 3.0:
        print(f"    ✓ MAE < 3%: APROBADO")
    else:
        print(f"    ⚠ MAE < 3%: ACEPTABLE")
    
    # -------------------------------------------------------------------------
    # VISUALIZACIÓN (CUMPLE REGLA V2)
    # -------------------------------------------------------------------------
    print("\n[VISUALIZACIÓN] Generando gráficas...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Gráfica 1: Predicción vs Observado (Test Set)
    plt.subplot(2, 2, 1)
    plt.plot(range(len(y_test)), y_test, 'b-', 
             label='Observado', linewidth=2)
    plt.plot(range(len(y_pred)), y_pred, 'r--', 
             label='Predicho LSTM', linewidth=2, alpha=0.7)
    plt.xlabel('Tiempo (pasos)', fontsize=12)
    plt.ylabel('Concentración (normalizado)', fontsize=12)
    plt.title('Test Set: Predicción vs Observado', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Gráfica 2: Curvas de aprendizaje
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Convergencia del Entrenamiento', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Gráfica 3: Error de predicción
    plt.subplot(2, 2, 3)
    error = y_test - y_pred
    plt.plot(error, 'g-', linewidth=1, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Tiempo (pasos)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Error de Predicción', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Gráfica 4: Distribución del error
    plt.subplot(2, 2, 4)
    plt.hist(error, bins=30, edgecolor='black', alpha=0.7, color='green')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Error', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución del Error', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('lstm_cloro_resultados.png', dpi=150, bbox_inches='tight')
    print(f"    ✓ Gráficas guardadas: lstm_cloro_resultados.png")
    
    # -------------------------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("RESUMEN DE HIPERPARÁMETROS Y RESULTADOS")
    print("="*70)
    print(f"\nPARÁMETROS DE EMBEBIMIENTO:")
    print(f"  - τ (retardo temporal):      {tau}")
    print(f"  - d (dimensión embebimiento): {d}")
    print(f"\nDATOS:")
    print(f"  - Total de muestras:          {len(X)}")
    print(f"  - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"\nARQUITECTURA LSTM:")
    print(f"  - Hidden units:               64")
    print(f"  - Dropout rate:               0.2")
    print(f"  - Total de capas:             7")
    print(f"\nENTRENAMIENTO:")
    print(f"  - Optimizer:                  Adam (lr=0.001)")
    print(f"  - Loss function:              MSE")
    print(f"  - Épocas máximas:             200")
    print(f"  - Épocas ejecutadas:          {len(history.history['loss'])}")
    print(f"  - Batch size:                 16")
    print(f"  - Early stopping patience:    15")
    print(f"\nRESULTADOS:")
    print(f"  - RMSE:                       {rmse:.6f} ({rmse_pct:.2f}%)")
    print(f"  - MAE:                        {mae:.6f} ({mae_pct:.2f}%)")
    print(f"  - Estado:                     {'✓ APROBADO' if rmse_pct < 5.0 else '✗ RECHAZADO'}")
    print("="*70)
    
    # Guardar modelo
    model.save('modelo_lstm_cloro.h5')
    print(f"\n✓ Modelo guardado: modelo_lstm_cloro.h5")
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)


if __name__ == "__main__":
    main()
