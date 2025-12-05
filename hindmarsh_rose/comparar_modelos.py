"""
COMPARACIÓN: MODELO ORIGINAL vs MODELO ESN
===========================================
Este script compara la serie temporal generada por el modelo físico
de Hindmarsh-Rose con las predicciones del modelo ESN entrenado.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class EchoStateNetwork:
    """Echo State Network para carga desde archivo"""
    
    def __init__(self, reservoir_size=300, spectral_radius=0.9, input_scale=1.0):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.W_res = None
        self.W_in = None
        self.W_out = None
    
    def _step_reservoir(self, h, x):
        """Un paso de la dinámica del reservorio"""
        return np.tanh(self.W_res @ h + self.W_in @ x)
    
    def _generate_states(self, X):
        """Genera estados del reservorio"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, _ = X.shape
        states = np.zeros((n_samples, self.reservoir_size))
        h = np.zeros(self.reservoir_size)
        
        for i in range(n_samples):
            h = self._step_reservoir(h, X[i])
            states[i] = h
        
        return states
    
    def predict_onestep(self, x):
        """Predicción one-step"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        H = self._generate_states(x)
        return (H @ self.W_out)[0]
    
    def predict_multistep(self, x_init, steps):
        """Predicción multi-step"""
        predictions = []
        h = np.zeros(self.reservoir_size)
        x = x_init.copy()
        
        for _ in range(steps):
            h = self._step_reservoir(h, x)
            y_next = self.W_out @ h
            predictions.append(y_next)
            
            x = np.roll(x, -1)
            x[-1] = y_next
        
        return np.array(predictions)


def hindmarsh_rose(state, t, a=1.0, b=3.0, c=1.0, d=5.0, r=0.001, s=4.0, x_r=-1.6, I=3.25):
    """Sistema de ecuaciones de Hindmarsh-Rose"""
    x, y, z = state
    
    dx_dt = y - a * x**3 + b * x**2 - z + I
    dy_dt = c - d * x**2 - y
    dz_dt = r * (s * (x - x_r) - z)
    
    return [dx_dt, dy_dt, dz_dt]


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
    print("COMPARACIÓN: MODELO FÍSICO vs MODELO ESN")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. CARGAR DATOS ORIGINALES
    # -------------------------------------------------------------------------
    print("\n[1/5] Cargando datos originales...")
    df = pd.read_csv('datos_hindmarsh_rose.csv')
    tiempo_original = df['tiempo'].values
    voltaje_original = df['voltaje'].values
    print(f"    ✓ {len(voltaje_original)} puntos cargados")
    
    # -------------------------------------------------------------------------
    # 2. NORMALIZAR DATOS
    # -------------------------------------------------------------------------
    print("\n[2/5] Normalizando datos...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie_norm = scaler.fit_transform(voltaje_original.reshape(-1, 1)).flatten()
    
    # -------------------------------------------------------------------------
    # 3. CARGAR MODELO PARA OBTENER PARÁMETROS
    # -------------------------------------------------------------------------
    print("\n[3/5] Cargando parámetros del modelo...")
    try:
        data = np.load('modelo_esn_hindmarsh_rose.npz')
        tau = int(data['tau'])
        d = int(data['d'])
        print(f"    ✓ Parámetros cargados: τ={tau}, d={d}")
    except FileNotFoundError:
        print(f"    ✗ Error: No se encontró 'modelo_esn_hindmarsh_rose.npz'")
        print("    → Ejecutar primero: modelo_hindmarsh_rose_esn.py")
        return
    
    # -------------------------------------------------------------------------
    # 4. PREPARAR DATOS PARA ESN (mismo proceso que entrenamiento)
    # -------------------------------------------------------------------------
    print("\n[4/5] Preparando datos...")
    
    X, y = crear_matriz_embebida(serie_norm, tau, d)
    
    # Split igual que en entrenamiento
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, shuffle=False
    )
    
    # Índices correspondientes en la serie original
    test_start_idx = len(X_trainval) + d * tau
    
    # -------------------------------------------------------------------------
    # 5. CARGAR MODELO Y HACER PREDICCIONES
    # -------------------------------------------------------------------------
    print("\n[5/5] Cargando modelo ESN y prediciendo...")
    try:
        # Reconstruir ESN con parámetros por defecto
        esn = EchoStateNetwork(
            reservoir_size=300,
            spectral_radius=0.9,
            input_scale=1.0
        )
        esn.W_res = data['W_res']
        esn.W_in = data['W_in']
        esn.W_out = data['W_out']
        
        print(f"    ✓ ESN cargada: {esn.reservoir_size} unidades, ρ={esn.spectral_radius}")
        
        # Predicciones one-step en test set
        y_pred_norm = np.array([esn.predict_onestep(x) for x in X_test])
        
        # Desnormalizar predicciones
        y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"    ✓ Predicciones one-step generadas: {len(y_pred)} puntos")
        
        # Calcular RMSE
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        rmse_percent = (rmse / np.std(y_true)) * 100
        print(f"    ✓ RMSE en test: {rmse:.6f} ({rmse_percent:.2f}%)")
        
    except Exception as e:
        print(f"    ✗ Error al cargar modelo: {e}")
        return
    
    # -------------------------------------------------------------------------
    # 6. GENERAR MODELO FÍSICO COMPLETO (para referencia)
    # -------------------------------------------------------------------------
    print("\n[6/6] Generando modelo físico de referencia...")
    
    # Parámetros del modelo físico
    estado_inicial = [-1.0, -5.0, 2.0]
    tiempo_ref = tiempo_original
    
    # Generar serie sin ruido (modelo puro)
    solucion_ref = odeint(hindmarsh_rose, estado_inicial, tiempo_ref)
    voltaje_ref = solucion_ref[:, 0]
    
    # -------------------------------------------------------------------------
    # 6. GENERAR PREDICCIONES PARA TODA LA SERIE
    # -------------------------------------------------------------------------
    print("\n[EXTRA] Generando predicciones para serie completa...")
    
    # Predicciones one-step para todo el conjunto
    y_pred_all_norm = np.array([esn.predict_onestep(x) for x in X])
    y_pred_all = scaler.inverse_transform(y_pred_all_norm.reshape(-1, 1)).flatten()
    
    # Tiempo correspondiente a las predicciones (ajustado por embebimiento)
    tiempo_pred_all = tiempo_original[d * tau:d * tau + len(y_pred_all)]
    
    # -------------------------------------------------------------------------
    # 7. PREDICCIÓN MULTI-STEP (horizonte corto para caos)
    # -------------------------------------------------------------------------
    print("\n[MULTI-STEP] Generando predicción autónoma...")
    
    # Punto de inicio: primer punto del test set
    x_inicial = X_test[0]
    n_steps = min(200, len(X_test))  # Predicción de 200 pasos
    
    y_multistep_norm = esn.predict_multistep(x_inicial, n_steps)
    y_multistep = scaler.inverse_transform(y_multistep_norm.reshape(-1, 1)).flatten()
    
    tiempo_multistep = tiempo_original[test_start_idx:test_start_idx + n_steps]
    y_true_multistep = voltaje_original[test_start_idx:test_start_idx + n_steps]
    
    # -------------------------------------------------------------------------
    # 8. CREAR VISUALIZACIONES COMPARATIVAS
    # -------------------------------------------------------------------------
    print("\n[VISUALIZACIÓN] Generando gráficas comparativas...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # ------------------------------------
    # Gráfica 1: Serie completa con TODAS las curvas
    # ------------------------------------
    plt.subplot(2, 2, 1)
    plt.plot(tiempo_original, voltaje_original, 'b-', 
             label='Datos Observados (con ruido)', linewidth=1, alpha=0.6)
    plt.plot(tiempo_ref, voltaje_ref, 'r--', 
             label='Modelo Físico Puro', linewidth=1.5, alpha=0.7)
    plt.plot(tiempo_pred_all, y_pred_all, 'g-', 
             label='Predicción ESN (one-step)', linewidth=1, alpha=0.8)
    
    # Marcar región de test
    test_time_start = tiempo_original[test_start_idx]
    plt.axvline(x=test_time_start, color='orange', linestyle=':', 
                linewidth=2, label=f'Inicio Test Set (t={test_time_start:.1f})')
    
    plt.xlabel('Tiempo (unidades)', fontsize=11)
    plt.ylabel('Voltaje (mV)', fontsize=11)
    plt.title('Serie Completa: Observado vs Físico vs ESN', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    
    # ------------------------------------
    # Gráfica 2: Zoom en región de test (one-step)
    # ------------------------------------
    plt.subplot(2, 2, 2)
    tiempo_test = tiempo_original[test_start_idx:test_start_idx + len(y_test)]
    
    # Mostrar solo primeros 500 puntos para claridad
    n_display = min(500, len(y_test))
    
    plt.plot(tiempo_test[:n_display], y_true[:n_display], 'b-', 
             label='Observado (Test Set)', linewidth=1.5, alpha=0.7)
    plt.plot(tiempo_test[:n_display], y_pred[:n_display], 'g--', 
             label='Predicción ESN (one-step)', linewidth=1.5, alpha=0.8)
    
    # Agregar modelo físico en la misma región
    plt.plot(tiempo_test[:n_display], voltaje_ref[test_start_idx:test_start_idx+n_display], 'r:', 
             label='Modelo Físico', linewidth=1.5, alpha=0.6)
    
    plt.xlabel('Tiempo (unidades)', fontsize=11)
    plt.ylabel('Voltaje (mV)', fontsize=11)
    plt.title(f'Test Set - One-Step (primeros {n_display} puntos)', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # ------------------------------------
    # Gráfica 3: Predicción Multi-Step (caos)
    # ------------------------------------
    plt.subplot(2, 2, 3)
    
    plt.plot(tiempo_multistep, y_true_multistep, 'b-', 
             label='Sistema Real', linewidth=2, alpha=0.7)
    plt.plot(tiempo_multistep, y_multistep, 'g--', 
             label='Predicción ESN (autónoma)', linewidth=2, alpha=0.8)
    
    plt.xlabel('Tiempo (unidades)', fontsize=11)
    plt.ylabel('Voltaje (mV)', fontsize=11)
    plt.title(f'Predicción Multi-Step ({n_steps} pasos)', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Añadir texto explicativo
    plt.text(0.05, 0.95, 'Sistema caótico: divergencia esperada', 
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ------------------------------------
    # Gráfica 4: Error absoluto en test set
    # ------------------------------------
    plt.subplot(2, 2, 4)
    error_abs = np.abs(y_pred - y_true)
    
    plt.plot(tiempo_test, error_abs, 'r-', linewidth=1.5, alpha=0.7)
    plt.fill_between(tiempo_test, 0, error_abs, alpha=0.3, color='red')
    
    # Estadísticas del error
    error_mean = np.mean(error_abs)
    error_std = np.std(error_abs)
    error_max = np.max(error_abs)
    
    plt.axhline(y=error_mean, color='orange', linestyle='--', 
                linewidth=2, label=f'Error Medio = {error_mean:.4f}')
    plt.axhline(y=error_mean + error_std, color='yellow', linestyle=':', 
                linewidth=1.5, label=f'Media + 1σ = {error_mean+error_std:.4f}')
    
    plt.xlabel('Tiempo (unidades)', fontsize=11)
    plt.ylabel('Error Absoluto (mV)', fontsize=11)
    plt.title('Distribución del Error (One-Step)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Añadir estadísticas
    stats_text = f'RMSE: {rmse:.4f}\nMáx: {error_max:.4f}\nσ: {error_std:.4f}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ------------------------------------
    # Ajustes finales
    # ------------------------------------
    plt.tight_layout()
    plt.savefig('comparacion_hindmarsh_rose.png', dpi=300, bbox_inches='tight')
    print("    ✓ Gráfica guardada: comparacion_hindmarsh_rose.png")
    
    # -------------------------------------------------------------------------
    # 9. REPORTE FINAL
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("REPORTE DE COMPARACIÓN")
    print("="*70)
    
    print("\n[CONFIGURACIÓN]")
    print(f"  • Delay de embebimiento (τ): {tau}")
    print(f"  • Dimensión de embebimiento (d): {d}")
    print(f"  • Puntos totales: {len(voltaje_original)}")
    print(f"  • Puntos en test: {len(y_test)}")
    print(f"  • Test set inicio: t = {test_time_start:.2f}")
    
    print("\n[MÉTRICAS ONE-STEP]")
    print(f"  • RMSE: {rmse:.6f} ({rmse_percent:.2f}%)")
    print(f"  • Error medio: {error_mean:.6f}")
    print(f"  • Error std: {error_std:.6f}")
    print(f"  • Error máximo: {error_max:.6f}")
    
    print("\n[MÉTRICAS MULTI-STEP]")
    error_multistep = np.abs(y_multistep - y_true_multistep)
    rmse_multistep = np.sqrt(np.mean(error_multistep**2))
    print(f"  • RMSE ({n_steps} pasos): {rmse_multistep:.6f}")
    print(f"  • Error medio: {np.mean(error_multistep):.6f}")
    print(f"  • Error máximo: {np.max(error_multistep):.6f}")
    
    # Calcular horizonte de predicción (error < 2*std)
    threshold = 2 * np.std(y_true_multistep)
    valid_steps = np.sum(error_multistep < threshold)
    print(f"  • Horizonte válido (error < 2σ): {valid_steps} pasos")
    
    print("\n[CARACTERÍSTICAS DEL SISTEMA]")
    print(f"  • Voltaje rango: [{np.min(voltaje_original):.4f}, {np.max(voltaje_original):.4f}]")
    print(f"  • Voltaje std: {np.std(voltaje_original):.4f}")
    print(f"  • Sistema: CAÓTICO (Hindmarsh-Rose)")
    
    print("\n[ARCHIVOS GENERADOS]")
    print(f"  ✓ comparacion_hindmarsh_rose.png")
    
    print("\n" + "="*70)
    print("COMPARACIÓN COMPLETADA")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
