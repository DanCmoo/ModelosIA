"""
GENERADOR DE DATOS: MODELO DE HINDMARSH-ROSE
=============================================
Este script genera la serie temporal del sistema de Hindmarsh-Rose
y la guarda en CSV. Solo se ejecuta UNA VEZ.

CUMPLE REGLA G1: Separación Caja Negra
- Las ecuaciones diferenciales SOLO existen aquí
- El modelo IA NUNCA verá estas ecuaciones
- Solo cargará el CSV generado

Sistema de Hindmarsh-Rose:
Modelo neuronal que exhibe comportamiento caótico
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt


def hindmarsh_rose(state, t, a=1.0, b=3.0, c=1.0, d=5.0, r=0.001, s=4.0, x_r=-1.6, I=3.25):
    """
    Sistema de ecuaciones de Hindmarsh-Rose
    
    Variables de estado:
    x : Potencial de membrana (variable observable)
    y : Tasa de recuperación
    z : Corriente lenta de adaptación
    
    Parámetros:
    a, b, c, d : Parámetros del modelo
    r : Escala temporal de z
    s : Acoplamiento entre x y z
    x_r : Punto de equilibrio
    I : Corriente externa (controla el comportamiento)
    """
    x, y, z = state
    
    dx_dt = y - a * x**3 + b * x**2 - z + I
    dy_dt = c - d * x**2 - y
    dz_dt = r * (s * (x - x_r) - z)
    
    return [dx_dt, dy_dt, dz_dt]


def generar_serie_hindmarsh_rose(n_puntos=5000, dt=0.1, seed=42):
    """
    Genera serie temporal de Hindmarsh-Rose
    
    Parámetros:
    -----------
    n_puntos : int
        Número de puntos temporales a generar
    dt : float
        Paso de tiempo
    seed : int
        Semilla para reproducibilidad
    
    Returns:
    --------
    tiempo : array
        Vector de tiempo
    x : array
        Potencial de membrana (variable observable)
    """
    np.random.seed(seed)
    
    # Condiciones iniciales
    x0 = -1.3
    y0 = 0.0
    z0 = 3.0
    state0 = [x0, y0, z0]
    
    # Vector de tiempo
    tiempo = np.linspace(0, n_puntos * dt, n_puntos)
    
    # Resolver sistema de EDOs
    solucion = odeint(hindmarsh_rose, state0, tiempo)
    
    # Extraer solo x (potencial de membrana - la variable observable)
    x = solucion[:, 0]
    
    # Agregar ruido muy pequeño (simulando mediciones)
    ruido = np.random.normal(0, 0.01, n_puntos)
    x_ruidoso = x + ruido
    
    return tiempo, x_ruidoso


def main():
    """
    Función principal: genera datos y guarda en CSV
    """
    print("="*70)
    print("GENERACIÓN DE DATOS: HINDMARSH-ROSE")
    print("="*70)
    
    # Generar serie temporal
    print("\n[1/3] Generando serie temporal caótica...")
    tiempo, voltaje = generar_serie_hindmarsh_rose(n_puntos=5000, dt=0.1)
    
    print(f"    - Puntos generados: {len(voltaje)}")
    print(f"    - Rango temporal: {tiempo[0]:.2f} - {tiempo[-1]:.2f} unidades")
    print(f"    - Voltaje mín: {voltaje.min():.4f}")
    print(f"    - Voltaje máx: {voltaje.max():.4f}")
    print(f"    - Media: {voltaje.mean():.4f}")
    print(f"    - Desviación estándar: {voltaje.std():.4f}")
    
    # Crear DataFrame
    print("\n[2/3] Creando DataFrame...")
    df = pd.DataFrame({
        'tiempo': tiempo,
        'voltaje': voltaje
    })
    
    # Guardar a CSV
    output_file = 'datos_hindmarsh_rose.csv'
    print(f"\n[3/3] Guardando en '{output_file}'...")
    df.to_csv(output_file, index=False)
    print(f"    ✓ Archivo guardado exitosamente")
    
    # Visualización
    print("\n[BONUS] Generando visualización...")
    fig = plt.figure(figsize=(14, 10))
    
    # Serie temporal completa
    plt.subplot(2, 2, 1)
    plt.plot(tiempo, voltaje, 'b-', linewidth=0.5)
    plt.xlabel('Tiempo', fontsize=11)
    plt.ylabel('Voltaje (x)', fontsize=11)
    plt.title('Serie Temporal Completa - Hindmarsh-Rose', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Zoom en primeros 200 puntos
    plt.subplot(2, 2, 2)
    plt.plot(tiempo[:200], voltaje[:200], 'r-', linewidth=1)
    plt.xlabel('Tiempo', fontsize=11)
    plt.ylabel('Voltaje (x)', fontsize=11)
    plt.title('Zoom: Primeros 200 puntos', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Histograma
    plt.subplot(2, 2, 3)
    plt.hist(voltaje, bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('Voltaje (x)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title('Distribución de Valores', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Diagrama de fase (x vs dx/dt aproximado)
    plt.subplot(2, 2, 4)
    dx_dt_approx = np.diff(voltaje) / np.diff(tiempo)
    plt.plot(voltaje[:-1], dx_dt_approx, 'purple', linewidth=0.3, alpha=0.6)
    plt.xlabel('Voltaje x(t)', fontsize=11)
    plt.ylabel('dx/dt (aproximado)', fontsize=11)
    plt.title('Diagrama de Fase', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('datos_hindmarsh_rose_visualizacion.png', dpi=150, bbox_inches='tight')
    print(f"    ✓ Visualización guardada como 'datos_hindmarsh_rose_visualizacion.png'")
    
    print("\n" + "="*70)
    print("GENERACIÓN COMPLETADA")
    print("="*70)
    print("\nCARACTERÍSTICAS DEL SISTEMA:")
    print("  - Tipo: Sistema caótico de 3 dimensiones")
    print("  - Variable observable: x (potencial de membrana)")
    print("  - Comportamiento: Spikes irregulares (bursting)")
    print("  - Horizonte de Lyapunov: ~5-15 pasos típicamente")
    print("\nPRÓXIMOS PASOS:")
    print("1. Ejecutar modelo_hindmarsh_rose_esn.py para entrenar ESN")
    print("2. El modelo IA SOLO verá el CSV, NO las ecuaciones")
    print("="*70)


if __name__ == "__main__":
    main()
