"""
EJECUTAR PIPELINE COMPLETO - HINDMARSH-ROSE
============================================
Script que ejecuta todo el proceso de generación y modelado ESN
"""

import subprocess
import sys

def main():
    print("="*70)
    print("PIPELINE COMPLETO: MODELO DE HINDMARSH-ROSE (ESN)")
    print("="*70)
    
    # Paso 1: Generar datos
    print("\n" + "="*70)
    print("ETAPA 1: GENERACIÓN DE DATOS")
    print("="*70)
    try:
        subprocess.run([sys.executable, "generar_datos_hindmarsh_rose.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error en generación de datos: {e}")
        return
    
    # Paso 2: Entrenar modelo ESN
    print("\n" + "="*70)
    print("ETAPA 2: ENTRENAMIENTO DEL MODELO ESN")
    print("="*70)
    try:
        subprocess.run([sys.executable, "modelo_hindmarsh_rose_esn.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error en entrenamiento: {e}")
        return
    
    # Paso 3: Comparar modelos
    print("\n" + "="*70)
    print("ETAPA 3: COMPARACIÓN DE MODELOS")
    print("="*70)
    try:
        subprocess.run([sys.executable, "comparar_modelos.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error en comparación: {e}")
        return
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("\nARCHIVOS GENERADOS:")
    print("  - datos_hindmarsh_rose.csv")
    print("  - datos_hindmarsh_rose_visualizacion.png")
    print("  - modelo_esn_hindmarsh_rose.npz")
    print("  - esn_hindmarsh_rose_resultados.png")
    print("  - comparacion_hindmarsh_rose.png")
    print("="*70)

if __name__ == "__main__":
    main()
