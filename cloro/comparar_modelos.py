"""
COMPARACIÓN: MODELO ORIGINAL vs MODELO LSTM
============================================
Este script compara la serie temporal generada por el modelo físico
con las predicciones del modelo LSTM entrenado.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


def modelo_cloro(C, t, k, C_entrada, tasa_flujo, volumen):
    """Ecuación diferencial del modelo de cloro"""
    dCdt = (tasa_flujo / volumen) * (C_entrada - C) - k * C
    return dCdt


def crear_matriz_embebida(serie, tau, d):
    """Crea matriz embebida según el Teorema de Takens"""
    n = len(serie)
    max_idx = n - d * tau
    
    X = np.zeros((max_idx, d))
    y = np.zeros(max_idx)
    
    for i in range(max_idx):
        for j in range(d):
            X[i, j] = serie[i + j * tau]
        y[i] = serie[i + d * tau]
    
    return X, y


def main():
    print("="*70)
    print("COMPARACIÓN: MODELO FÍSICO vs MODELO LSTM")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. CARGAR DATOS ORIGINALES
    # -------------------------------------------------------------------------
    print("\n[1/5] Cargando datos originales...")
    df = pd.read_csv('datos_cloro.csv')
    tiempo_original = df['tiempo'].values
    concentracion_original = df['concentracion'].values
    print(f"    ✓ {len(concentracion_original)} puntos cargados")
    
    # -------------------------------------------------------------------------
    # 2. NORMALIZAR DATOS
    # -------------------------------------------------------------------------
    print("\n[2/5] Normalizando datos...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie_norm = scaler.fit_transform(concentracion_original.reshape(-1, 1)).flatten()
    
    # -------------------------------------------------------------------------
    # 3. PREPARAR DATOS PARA LSTM (mismo proceso que entrenamiento)
    # -------------------------------------------------------------------------
    print("\n[3/5] Preparando datos...")
    tau = 12
    d = 3
    
    X, y = crear_matriz_embebida(serie_norm, tau, d)
    
    # Split igual que en entrenamiento
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    
    # Índices correspondientes en la serie original
    test_start_idx = len(X_trainval) + d * tau
    
    # -------------------------------------------------------------------------
    # 4. CARGAR MODELO Y HACER PREDICCIONES
    # -------------------------------------------------------------------------
    print("\n[4/5] Cargando modelo y prediciendo...")
    try:
        # Cargar modelo con compile=False para evitar problemas de serialización
        model = keras.models.load_model('modelo_lstm_cloro.h5', compile=False)
        
        # Recompilar el modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        y_pred_norm = model.predict(X_test, verbose=0).flatten()
        
        # Desnormalizar predicciones
        y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"    ✓ Predicciones generadas: {len(y_pred)} puntos")
    except Exception as e:
        print(f"    ✗ Error al cargar modelo: {e}")
        print("    → Ejecutar primero: modelo_cloro_lstm.py")
        return
    
    # -------------------------------------------------------------------------
    # 5. GENERAR MODELO FÍSICO COMPLETO (para referencia)
    # -------------------------------------------------------------------------
    print("\n[5/5] Generando modelo físico de referencia...")
    
    # Parámetros del modelo físico
    k = 0.15
    C_entrada = 2.0
    tasa_flujo = 100.0
    volumen = 1000.0
    C0 = 0.5
    
    # Generar serie sin ruido (modelo puro)
    tiempo_ref = np.linspace(0, 100, 1000)
    concentracion_ref = odeint(
        modelo_cloro, 
        C0, 
        tiempo_ref, 
        args=(k, C_entrada, tasa_flujo, volumen)
    ).flatten()
    
    # -------------------------------------------------------------------------
    # 6. GENERAR PREDICCIONES PARA TODA LA SERIE
    # -------------------------------------------------------------------------
    print("\n[EXTRA] Generando predicciones para serie completa...")
    
    # Predicciones para todo el conjunto (train + val + test)
    y_pred_all_norm = model.predict(X, verbose=0).flatten()
    y_pred_all = scaler.inverse_transform(y_pred_all_norm.reshape(-1, 1)).flatten()
    
    # Tiempo correspondiente a las predicciones (ajustado por embebimiento)
    tiempo_pred_all = tiempo_original[d * tau:d * tau + len(y_pred_all)]
    
    # -------------------------------------------------------------------------
    # 7. CREAR VISUALIZACIONES COMPARATIVAS
    # -------------------------------------------------------------------------
    print("\n[VISUALIZACIÓN] Generando gráficas comparativas...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # ------------------------------------
    # Gráfica 1: Serie completa con TODAS las curvas
    # ------------------------------------
    plt.subplot(2, 2, 1)
    plt.plot(tiempo_original, concentracion_original, 'b-', 
             label='Datos Observados (con ruido)', linewidth=1.5, alpha=0.6)
    plt.plot(tiempo_ref, concentracion_ref, 'r--', 
             label='Modelo Físico Puro', linewidth=2, alpha=0.7)
    plt.plot(tiempo_pred_all, y_pred_all, 'g-', 
             label='Predicción LSTM (toda la serie)', linewidth=1.5, alpha=0.8)
    
    # Marcar región de test
    test_time_start = tiempo_original[test_start_idx]
    plt.axvline(x=test_time_start, color='orange', linestyle=':', 
                linewidth=2, label=f'Inicio Test Set (t={test_time_start:.1f}h)')
    
    plt.xlabel('Tiempo (horas)', fontsize=11)
    plt.ylabel('Concentración (mg/L)', fontsize=11)
    plt.title('Serie Completa: Observado vs Físico vs LSTM', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    
    # ------------------------------------
    # Gráfica 2: Zoom en región de test
    # ------------------------------------
    plt.subplot(2, 2, 2)
    tiempo_test = tiempo_original[test_start_idx:test_start_idx + len(y_test)]
    
    plt.plot(tiempo_test, y_true, 'b-', 
             label='Observado (Test Set)', linewidth=2, marker='o', 
             markersize=3, alpha=0.7)
    plt.plot(tiempo_test, y_pred, 'g--', 
             label='Predicción LSTM', linewidth=2, marker='s', 
             markersize=3, alpha=0.8)
    
    # Agregar modelo físico en la misma región
    idx_ref_start = int(tiempo_test[0] / 100 * len(tiempo_ref))
    idx_ref_end = idx_ref_start + len(tiempo_test)
    plt.plot(tiempo_test, concentracion_ref[idx_ref_start:idx_ref_end], 'r:', 
             label='Modelo Físico', linewidth=2, alpha=0.6)
    
    plt.xlabel('Tiempo (horas)', fontsize=11)
    plt.ylabel('Concentración (mg/L)', fontsize=11)
    plt.title('Comparación en Test Set', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # ------------------------------------
    # Gráfica 3: Superpuesta - Primeros 100 puntos del test
    # ------------------------------------
    plt.subplot(2, 2, 3)
    n_points = min(100, len(y_test))
    tiempo_zoom = tiempo_test[:n_points]
    
    plt.plot(tiempo_zoom, y_true[:n_points], 'b-', 
             label='Observado', linewidth=2.5, marker='o', 
             markersize=4, alpha=0.7)
    plt.plot(tiempo_zoom, y_pred[:n_points], 'g--', 
             label='Predicción LSTM', linewidth=2.5, marker='s', 
             markersize=4, alpha=0.8)
    
    plt.xlabel('Tiempo (horas)', fontsize=11)
    plt.ylabel('Concentración (mg/L)', fontsize=11)
    plt.title(f'Zoom: Primeros {n_points} puntos del Test', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # ------------------------------------
    # Gráfica 4: Error absoluto
    # ------------------------------------
    plt.subplot(2, 2, 4)
    error = np.abs(y_true - y_pred)
    plt.plot(tiempo_test, error, 'purple', linewidth=1.5)
    plt.axhline(y=np.mean(error), color='r', linestyle='--', 
                linewidth=2, label=f'Error Medio: {np.mean(error):.4f} mg/L')
    plt.fill_between(tiempo_test, 0, error, alpha=0.3, color='purple')
    plt.xlabel('Tiempo (horas)', fontsize=11)
    plt.ylabel('Error Absoluto (mg/L)', fontsize=11)
    plt.title('Error de Predicción LSTM', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacion_modelo_fisico_vs_lstm.png', dpi=150, bbox_inches='tight')
    print(f"    ✓ Gráfica guardada: comparacion_modelo_fisico_vs_lstm.png")
    
    # -------------------------------------------------------------------------
    # 8. ESTADÍSTICAS COMPARATIVAS
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("ESTADÍSTICAS COMPARATIVAS")
    print("="*70)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMÉTRICAS DE ERROR:")
    print(f"  - RMSE:           {rmse:.6f} mg/L")
    print(f"  - MAE:            {mae:.6f} mg/L")
    print(f"  - MAPE:           {mape:.2f}%")
    print(f"  - R²:             {r2:.6f}")
    
    print(f"\nESTADÍSTICAS DESCRIPTIVAS:")
    print(f"\n  Observado (Test):")
    print(f"    - Media:        {np.mean(y_true):.6f} mg/L")
    print(f"    - Std Dev:      {np.std(y_true):.6f} mg/L")
    print(f"    - Min:          {np.min(y_true):.6f} mg/L")
    print(f"    - Max:          {np.max(y_true):.6f} mg/L")
    
    print(f"\n  Predicho (LSTM):")
    print(f"    - Media:        {np.mean(y_pred):.6f} mg/L")
    print(f"    - Std Dev:      {np.std(y_pred):.6f} mg/L")
    print(f"    - Min:          {np.min(y_pred):.6f} mg/L")
    print(f"    - Max:          {np.max(y_pred):.6f} mg/L")
    
    print(f"\n  Diferencias:")
    print(f"    - Δ Media:      {abs(np.mean(y_true) - np.mean(y_pred)):.6f} mg/L")
    print(f"    - Δ Std:        {abs(np.std(y_true) - np.std(y_pred)):.6f} mg/L")
    
    print("\n" + "="*70)
    print("COMPARACIÓN COMPLETADA")
    print("="*70)
    print("\nINTERPRETACIÓN:")
    print(f"  - R² = {r2:.4f}: El modelo LSTM explica el {r2*100:.1f}% de la varianza")
    print(f"  - MAE = {mae:.4f} mg/L: Error promedio absoluto por predicción")
    print(f"  - MAPE = {mape:.2f}%: Error porcentual promedio")
    
    if r2 > 0.95:
        print("\n  ✓ EXCELENTE: El modelo captura muy bien la dinámica")
    elif r2 > 0.90:
        print("\n  ✓ BUENO: El modelo captura bien la dinámica")
    elif r2 > 0.80:
        print("\n  ⚠ ACEPTABLE: El modelo captura la tendencia general")
    else:
        print("\n  ✗ MEJORABLE: Considerar ajustar hiperparámetros")
    
    print("="*70)
    
    # Mostrar gráficas en ventana
    plt.show()


if __name__ == "__main__":
    main()
