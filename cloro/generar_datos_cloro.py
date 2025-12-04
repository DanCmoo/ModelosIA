"""
GENERADOR DE DATOS: CONCENTRACIÓN DE CLORO
==========================================
Este script genera la serie temporal de concentración de cloro
y la guarda en CSV. Solo se ejecuta UNA VEZ.

CUMPLE REGLA G1: Separación Caja Negra
- Las ecuaciones diferenciales SOLO existen aquí
- El modelo IA NUNCA verá estas ecuaciones
- Solo cargará el CSV generado
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt


def modelo_cloro(C, t, k, C_entrada, tasa_flujo, volumen):
    """
    Ecuación diferencial del modelo de cloro en un tanque
    
    dC/dt = (tasa_flujo/volumen) * (C_entrada - C) - k*C
    
    Parámetros:
    -----------
    C : float
        Concentración actual de cloro (mg/L)
    t : float
        Tiempo (horas)
    k : float
        Constante de descomposición del cloro (1/hora)
    C_entrada : float
        Concentración de cloro en la entrada (mg/L)
    tasa_flujo : float
        Tasa de flujo de agua (L/hora)
    volumen : float
        Volumen del tanque (L)
    
    Returns:
    --------
    dCdt : float
        Tasa de cambio de concentración
    """
    dCdt = (tasa_flujo / volumen) * (C_entrada - C) - k * C
    return dCdt


def generar_serie_cloro(n_puntos=1000, dt=0.1, seed=42):
    """
    Genera serie temporal de concentración de cloro
    
    Parámetros:
    -----------
    n_puntos : int
        Número de puntos temporales a generar
    dt : float
        Paso de tiempo (horas)
    seed : int
        Semilla para reproducibilidad
    
    Returns:
    --------
    tiempo : array
        Vector de tiempo
    concentracion : array
        Serie temporal de concentración
    """
    np.random.seed(seed)
    
    # Parámetros del sistema
    k = 0.15                    # Constante descomposición (1/hora)
    C_entrada = 2.0             # Concentración entrada (mg/L)
    tasa_flujo = 100.0          # Tasa flujo (L/hora)
    volumen = 1000.0            # Volumen tanque (L)
    
    # Condición inicial
    C0 = 0.5  # mg/L
    
    # Vector de tiempo
    tiempo = np.linspace(0, n_puntos * dt, n_puntos)
    
    # Resolver EDO
    concentracion = odeint(
        modelo_cloro, 
        C0, 
        tiempo, 
        args=(k, C_entrada, tasa_flujo, volumen)
    ).flatten()
    
    # Agregar ruido realista (variaciones de medición)
    ruido = np.random.normal(0, 0.02, n_puntos)
    concentracion_ruidosa = concentracion + ruido
    
    # Asegurar que no haya valores negativos
    concentracion_ruidosa = np.maximum(concentracion_ruidosa, 0)
    
    return tiempo, concentracion_ruidosa


def main():
    """
    Función principal: genera datos y guarda en CSV
    """
    print("="*60)
    print("GENERACIÓN DE DATOS: CONCENTRACIÓN DE CLORO")
    print("="*60)
    
    # Generar serie temporal
    print("\n[1/3] Generando serie temporal...")
    tiempo, concentracion = generar_serie_cloro(n_puntos=1000, dt=0.1)
    
    print(f"    - Puntos generados: {len(concentracion)}")
    print(f"    - Rango temporal: {tiempo[0]:.2f} - {tiempo[-1]:.2f} horas")
    print(f"    - Concentración mín: {concentracion.min():.4f} mg/L")
    print(f"    - Concentración máx: {concentracion.max():.4f} mg/L")
    print(f"    - Media: {concentracion.mean():.4f} mg/L")
    print(f"    - Desviación estándar: {concentracion.std():.4f} mg/L")
    
    # Crear DataFrame
    print("\n[2/3] Creando DataFrame...")
    df = pd.DataFrame({
        'tiempo': tiempo,
        'concentracion': concentracion
    })
    
    # Guardar a CSV
    output_file = 'datos_cloro.csv'
    print(f"\n[3/3] Guardando en '{output_file}'...")
    df.to_csv(output_file, index=False)
    print(f"    ✓ Archivo guardado exitosamente")
    
    # Visualización
    print("\n[BONUS] Generando visualización...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(tiempo, concentracion, 'b-', linewidth=1.5)
    plt.xlabel('Tiempo (horas)', fontsize=12)
    plt.ylabel('Concentración (mg/L)', fontsize=12)
    plt.title('Serie Temporal Completa', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(concentracion, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Concentración (mg/L)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Valores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('datos_cloro_visualizacion.png', dpi=150, bbox_inches='tight')
    print(f"    ✓ Visualización guardada como 'datos_cloro_visualizacion.png'")
    
    print("\n" + "="*60)
    print("GENERACIÓN COMPLETADA")
    print("="*60)
    print("\nPRÓXIMOS PASOS:")
    print("1. Ejecutar modelo_cloro_lstm.py para entrenar el modelo IA")
    print("2. El modelo IA SOLO verá el CSV, NO las ecuaciones")
    print("="*60)


if __name__ == "__main__":
    main()
