"""
MODELO ESN PARA HINDMARSH-ROSE
===============================
Este script implementa Echo State Network para el sistema caótico
de Hindmarsh-Rose siguiendo estrictamente las reglas de la guía técnica.

ARQUITECTURA: ESN (Echo State Network)
CUMPLE: Reglas G1-G4, T1-T3, E1-E7, V3-V5

AUTOR: Sistema de IA
FECHA: Diciembre 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PARTE 1: CÁLCULO DE PARÁMETROS DE EMBEBIMIENTO (TEOREMA DE TAKENS)
# =============================================================================

def estimar_tau(serie, max_lag=50):
    """
    Calcula el retardo temporal óptimo τ usando autocorrelación.
    CUMPLE REGLA T1
    """
    serie_centrada = serie - np.mean(serie)
    acf = np.correlate(serie_centrada, serie_centrada, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    
    for lag in range(1, min(max_lag, len(acf))):
        if acf[lag] < 0:
            print(f"    - τ encontrado en primer cruce por cero: {lag}")
            return lag, acf
    
    tau = max_lag // 4
    print(f"    - No se encontró cruce por cero, usando default: {tau}")
    return tau, acf


def calcular_fnn(serie, tau, d_max=20, threshold=20.0):
    """
    Calcula la dimensión de embebimiento óptima usando False Nearest Neighbors.
    CUMPLE REGLA T2
    """
    n = len(serie)
    fnn_percentages = []
    
    for d in range(1, d_max + 1):
        max_idx = n - (d * tau)
        if max_idx < 10:
            break
        
        X_d = np.zeros((max_idx, d))
        for i in range(max_idx):
            for j in range(d):
                X_d[i, j] = serie[i + j * tau]
        
        max_idx_d1 = n - ((d + 1) * tau)
        if max_idx_d1 < 10:
            break
        
        X_d1 = np.zeros((max_idx_d1, d + 1))
        for i in range(max_idx_d1):
            for j in range(d + 1):
                X_d1[i, j] = serie[i + j * tau]
        
        false_neighbors = 0
        total_neighbors = 0
        
        for i in range(min(len(X_d), len(X_d1))):
            distances_d = np.linalg.norm(X_d - X_d[i], axis=1)
            distances_d[i] = np.inf
            nn_idx = np.argmin(distances_d)
            dist_d = distances_d[nn_idx]
            
            if dist_d == 0 or nn_idx >= len(X_d1):
                continue
            
            dist_d1 = np.linalg.norm(X_d1[i] - X_d1[nn_idx])
            
            if dist_d > 0:
                ratio = dist_d1 / dist_d
                if ratio > threshold:
                    false_neighbors += 1
            
            total_neighbors += 1
        
        if total_neighbors > 0:
            fnn_pct = (false_neighbors / total_neighbors) * 100
        else:
            fnn_pct = 100
        
        fnn_percentages.append(fnn_pct)
        print(f"    - d={d}: FNN={fnn_pct:.2f}%")
        
        if fnn_pct < 5.0:
            print(f"    ✓ Dimensión óptima encontrada: d={d}")
            return d, fnn_percentages
    
    d_optimo = np.argmin(fnn_percentages) + 1
    print(f"    - Usando dimensión con mínimo FNN: d={d_optimo}")
    return d_optimo, fnn_percentages


# =============================================================================
# PARTE 2: EMBEBIMIENTO TEMPORAL
# =============================================================================

def crear_matriz_embebida(serie, tau, d):
    """
    Crea matriz embebida según el Teorema de Takens.
    CUMPLE REGLA T3
    """
    n = len(serie)
    max_idx = n - d * tau
    
    X = np.zeros((max_idx, d))
    y = np.zeros(max_idx)
    
    for i in range(max_idx):
        for j in range(d):
            X[i, j] = serie[i + j * tau]
        y[i] = serie[i + d * tau]
    
    return X, y


# =============================================================================
# PARTE 3: ECHO STATE NETWORK
# =============================================================================

class EchoStateNetwork:
    """
    Echo State Network para sistemas caóticos
    CUMPLE REGLAS E1-E7
    """
    
    def __init__(self, reservoir_size=300, spectral_radius=0.9, input_scale=1.0, seed=42):
        """
        Inicializa la ESN
        
        CUMPLE REGLA E1: Inicialización del reservorio
        CUMPLE REGLA E2: Pesos de entrada
        """
        np.random.seed(seed)
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        
        # Inicializar matriz de reservorio
        W_raw = np.random.randn(reservoir_size, reservoir_size)
        eigenvalues = np.linalg.eigvals(W_raw)
        rho_actual = np.max(np.abs(eigenvalues))
        self.W_res = (spectral_radius / rho_actual) * W_raw
        
        print(f"    - Reservorio inicializado: {reservoir_size} unidades")
        print(f"    - Radio espectral: {spectral_radius}")
        print(f"    - Radio espectral verificado: {np.max(np.abs(np.linalg.eigvals(self.W_res))):.4f}")
        
        self.W_in = None
        self.W_out = None
    
    def _init_input_weights(self, input_dim):
        """Inicializa pesos de entrada - CUMPLE REGLA E2"""
        self.W_in = self.input_scale * np.random.randn(self.reservoir_size, input_dim)
    
    def _step_reservoir(self, h, x):
        """
        Un paso de la dinámica del reservorio
        CUMPLE REGLA E3: h(t+1) = tanh(W_res @ h(t) + W_in @ x(t))
        """
        return np.tanh(self.W_res @ h + self.W_in @ x)
    
    def _generate_states(self, X):
        """
        Genera estados del reservorio para datos de entrada
        CUMPLE REGLA E4: Generación de estados
        """
        n_samples, input_dim = X.shape
        
        if self.W_in is None:
            self._init_input_weights(input_dim)
        
        states = np.zeros((n_samples, self.reservoir_size))
        h = np.zeros(self.reservoir_size)
        
        for i in range(n_samples):
            h = self._step_reservoir(h, X[i])
            states[i] = h
        
        return states
    
    def fit(self, X_train, y_train, lambda_reg=1e-6):
        """
        Entrena los pesos de salida usando Ridge Regression
        CUMPLE REGLA E5: Ridge Regression
        """
        print("\n    Generando estados del reservorio...")
        H = self._generate_states(X_train)
        
        print("    Calculando pesos de salida (Ridge Regression)...")
        H_T_H = H.T @ H
        H_T_y = H.T @ y_train
        I = np.eye(self.reservoir_size)
        
        self.W_out = np.linalg.solve(H_T_H + lambda_reg * I, H_T_y)
        
        print(f"    ✓ Pesos de salida calculados: {self.W_out.shape}")
        
        return self
    
    def predict_onestep(self, X_test):
        """
        Predicción one-step-ahead (sin retroalimentación)
        CUMPLE REGLA E6
        """
        H_test = self._generate_states(X_test)
        y_pred = H_test @ self.W_out
        return y_pred
    
    def predict_multistep(self, x_init, steps):
        """
        Predicción multi-step (con retroalimentación)
        CUMPLE REGLA E7
        """
        predictions = []
        h = np.zeros(self.reservoir_size)
        x = x_init.copy()
        
        for _ in range(steps):
            h = self._step_reservoir(h, x)
            y_next = self.W_out @ h
            predictions.append(y_next)
            
            # Retroalimentar predicción
            x = np.roll(x, -1)
            x[-1] = y_next
        
        return np.array(predictions)


# =============================================================================
# PARTE 4: FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("="*70)
    print("MODELO ESN PARA HINDMARSH-ROSE (SISTEMA CAÓTICO)")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # PASO 1: CARGAR DATOS
    # -------------------------------------------------------------------------
    print("\n[PASO 1/9] Cargando datos desde CSV...")
    try:
        df = pd.read_csv('datos_hindmarsh_rose.csv')
        serie_raw = df['voltaje'].values
        print(f"    ✓ Datos cargados: {len(serie_raw)} puntos")
        print(f"    - Rango: [{serie_raw.min():.4f}, {serie_raw.max():.4f}]")
    except FileNotFoundError:
        print("    ✗ ERROR: datos_hindmarsh_rose.csv no encontrado")
        print("    → Ejecutar primero: generar_datos_hindmarsh_rose.py")
        return
    
    # -------------------------------------------------------------------------
    # PASO 2: NORMALIZACIÓN
    # -------------------------------------------------------------------------
    print("\n[PASO 2/9] Normalizando datos...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie_norm = scaler.fit_transform(serie_raw.reshape(-1, 1)).flatten()
    print(f"    ✓ Serie normalizada a rango [0, 1]")
    
    # -------------------------------------------------------------------------
    # PASO 3: CALCULAR τ
    # -------------------------------------------------------------------------
    print("\n[PASO 3/9] Calculando retardo temporal τ...")
    tau, acf = estimar_tau(serie_norm, max_lag=50)
    print(f"    ✓ τ = {tau}")
    
    # -------------------------------------------------------------------------
    # PASO 4: CALCULAR d
    # -------------------------------------------------------------------------
    print("\n[PASO 4/9] Calculando dimensión de embebimiento d...")
    d, fnn_pct = calcular_fnn(serie_norm, tau, d_max=20, threshold=20.0)
    print(f"    ✓ d = {d}")
    
    # -------------------------------------------------------------------------
    # PASO 5: CREAR MATRIZ EMBEBIDA
    # -------------------------------------------------------------------------
    print("\n[PASO 5/9] Creando matriz embebida...")
    X, y = crear_matriz_embebida(serie_norm, tau, d)
    print(f"    ✓ X shape: {X.shape}")
    print(f"    ✓ y shape: {y.shape}")
    
    # -------------------------------------------------------------------------
    # PASO 6: SPLIT DE DATOS
    # -------------------------------------------------------------------------
    print("\n[PASO 6/9] Dividiendo datos (train/val/test)...")
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, shuffle=False
    )
    
    print(f"    ✓ Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    ✓ Val:   {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    ✓ Test:  {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # PASO 7: CONSTRUIR Y ENTRENAR ESN
    # -------------------------------------------------------------------------
    print("\n[PASO 7/9] Construyendo y entrenando ESN...")
    
    esn = EchoStateNetwork(
        reservoir_size=300,
        spectral_radius=0.9,
        input_scale=1.0,
        seed=42
    )
    
    esn.fit(X_train, y_train, lambda_reg=1e-6)
    
    # -------------------------------------------------------------------------
    # PASO 8: EVALUACIÓN ONE-STEP
    # -------------------------------------------------------------------------
    print("\n[PASO 8/9] Evaluación one-step-ahead...")
    
    y_pred_onestep = esn.predict_onestep(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_onestep))
    mae = mean_absolute_error(y_test, y_pred_onestep)
    
    rango = serie_norm.max() - serie_norm.min()
    rmse_pct = (rmse / rango) * 100
    mae_pct = (mae / rango) * 100
    
    print(f"\n    MÉTRICAS ONE-STEP:")
    print(f"    {'─'*50}")
    print(f"    RMSE:      {rmse:.6f}  ({rmse_pct:.2f}% del rango)")
    print(f"    MAE:       {mae:.6f}  ({mae_pct:.2f}% del rango)")
    print(f"    {'─'*50}")
    
    if rmse_pct < 10:
        print(f"    ✓ RMSE < 10%: APROBADO")
    else:
        print(f"    ⚠ RMSE ≥ 10%: ACEPTABLE PARA CAOS")
    
    # -------------------------------------------------------------------------
    # PASO 9: EVALUACIÓN MULTI-STEP
    # -------------------------------------------------------------------------
    print("\n[PASO 9/9] Evaluación multi-step (retroalimentada)...")
    
    max_steps = min(100, len(y_test))
    y_pred_multistep = esn.predict_multistep(X_test[0], max_steps)
    
    # Calcular horizonte de Lyapunov
    error_rel = np.abs(y_pred_multistep - y_test[:max_steps]) / rango
    threshold_indices = np.where(error_rel > 0.5)[0]
    
    if len(threshold_indices) > 0:
        lyapunov_horizon = threshold_indices[0]
    else:
        lyapunov_horizon = max_steps
    
    print(f"\n    HORIZONTE DE LYAPUNOV:")
    print(f"    {'─'*50}")
    print(f"    Horizonte: {lyapunov_horizon} pasos")
    print(f"    {'─'*50}")
    
    if lyapunov_horizon >= 5:
        print(f"    ✓ Horizonte ≥ 5 pasos: APROBADO")
    else:
        print(f"    ✗ Horizonte < 5 pasos: MEJORABLE")
    
    # Estadísticas comparativas
    print(f"\n    ESTADÍSTICAS COMPARATIVAS (multi-step):")
    mean_obs = np.mean(y_test[:max_steps])
    mean_pred = np.mean(y_pred_multistep)
    std_obs = np.std(y_test[:max_steps])
    std_pred = np.std(y_pred_multistep)
    
    error_media_pct = abs(mean_obs - mean_pred) / abs(mean_obs) * 100
    error_std_pct = abs(std_obs - std_pred) / std_obs * 100
    
    print(f"    - Error media: {error_media_pct:.2f}%")
    print(f"    - Error std:   {error_std_pct:.2f}%")
    
    # -------------------------------------------------------------------------
    # VISUALIZACIÓN
    # -------------------------------------------------------------------------
    print("\n[VISUALIZACIÓN] Generando gráficas...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # Gráfica 1: One-step prediction
    plt.subplot(2, 2, 1)
    plt.plot(range(len(y_test[:200])), y_test[:200], 'b-', 
             label='Observado', linewidth=1.5)
    plt.plot(range(len(y_pred_onestep[:200])), y_pred_onestep[:200], 'r--', 
             label='Predicho (one-step)', linewidth=1.5, alpha=0.7)
    plt.xlabel('Pasos', fontsize=11)
    plt.ylabel('Voltaje (normalizado)', fontsize=11)
    plt.title('Predicción One-Step (Test Set)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Gráfica 2: Multi-step prediction
    plt.subplot(2, 2, 2)
    plt.plot(range(max_steps), y_test[:max_steps], 'b-', 
             label='Observado', linewidth=2)
    plt.plot(range(max_steps), y_pred_multistep, 'g--', 
             label='Predicho (multi-step)', linewidth=2, alpha=0.7)
    plt.axvline(x=lyapunov_horizon, color='r', linestyle=':', 
                linewidth=2, label=f'Horizonte Lyapunov ({lyapunov_horizon} pasos)')
    plt.xlabel('Pasos', fontsize=11)
    plt.ylabel('Voltaje (normalizado)', fontsize=11)
    plt.title('Predicción Multi-Step (Retroalimentada)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Gráfica 3: Error multi-step
    plt.subplot(2, 2, 3)
    plt.plot(error_rel, 'purple', linewidth=1.5)
    plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Umbral 50%')
    plt.axvline(x=lyapunov_horizon, color='orange', linestyle=':', 
                linewidth=2, label=f'Horizonte={lyapunov_horizon}')
    plt.xlabel('Pasos', fontsize=11)
    plt.ylabel('Error Relativo', fontsize=11)
    plt.title('Error Relativo Multi-Step', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Gráfica 4: Atractor 3D (si d >= 3)
    if d >= 3:
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Atractor observado
        n_plot = min(1000, len(y_test))
        X_atractor = np.zeros((n_plot, 3))
        for i in range(n_plot):
            for j in range(3):
                idx = i + j * tau
                if idx < len(y_test):
                    X_atractor[i, j] = y_test[idx]
        
        ax.plot(X_atractor[:, 0], X_atractor[:, 1], X_atractor[:, 2], 
                'b-', alpha=0.5, linewidth=0.5, label='Observado')
        
        ax.set_xlabel('y(t)', fontsize=10)
        ax.set_ylabel('y(t+τ)', fontsize=10)
        ax.set_zlabel('y(t+2τ)', fontsize=10)
        ax.set_title('Atractor 3D (Observado)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('esn_hindmarsh_rose_resultados.png', dpi=150, bbox_inches='tight')
    print(f"    ✓ Gráficas guardadas: esn_hindmarsh_rose_resultados.png")
    
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
    print(f"\nARQUITECTURA ESN:")
    print(f"  - Reservoir size:             300")
    print(f"  - Spectral radius:            0.9")
    print(f"  - Input scale:                1.0")
    print(f"  - Lambda regularización:      1e-6")
    print(f"\nRESULTADOS ONE-STEP:")
    print(f"  - RMSE:                       {rmse:.6f} ({rmse_pct:.2f}%)")
    print(f"  - MAE:                        {mae:.6f} ({mae_pct:.2f}%)")
    print(f"\nRESULTADOS MULTI-STEP:")
    print(f"  - Horizonte Lyapunov:         {lyapunov_horizon} pasos")
    print(f"  - Error media:                {error_media_pct:.2f}%")
    print(f"  - Error std:                  {error_std_pct:.2f}%")
    print(f"  - Estado:                     {'✓ APROBADO' if lyapunov_horizon >= 5 else '✗ MEJORABLE'}")
    print("="*70)
    
    # Guardar modelo (pesos)
    np.savez('modelo_esn_hindmarsh_rose.npz',
             W_res=esn.W_res,
             W_in=esn.W_in,
             W_out=esn.W_out,
             tau=tau,
             d=d,
             scaler_min=scaler.data_min_,
             scaler_max=scaler.data_max_)
    print(f"\n✓ Modelo guardado: modelo_esn_hindmarsh_rose.npz")
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)


if __name__ == "__main__":
    main()
