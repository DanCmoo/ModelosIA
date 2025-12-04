"""
EJECUTAR PIPELINE COMPLETO
===========================
Script que ejecuta todo el proceso de generación y modelado
"""

import subprocess
import sys

def main():
    print("="*70)
    print("PIPELINE COMPLETO: MODELO DE CONCENTRACIÓN DE CLORO")
    print("="*70)
    
    # Paso 1: Generar datos
    print("\n" + "="*70)
    print("ETAPA 1: GENERACIÓN DE DATOS")
    print("="*70)
    try:
        subprocess.run([sys.executable, "generar_datos_cloro.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error en generación de datos: {e}")
        return
    
    # Paso 2: Entrenar modelo
    print("\n" + "="*70)
    print("ETAPA 2: ENTRENAMIENTO DEL MODELO LSTM")
    print("="*70)
    try:
        subprocess.run([sys.executable, "modelo_cloro_lstm.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error en entrenamiento: {e}")
        return
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("\nARCHIVOS GENERADOS:")
    print("  - datos_cloro.csv")
    print("  - datos_cloro_visualizacion.png")
    print("  - modelo_lstm_cloro.h5")
    print("  - lstm_cloro_resultados.png")
    print("="*70)

if __name__ == "__main__":
    main()
