# Instrucciones de Instalación y Uso con UV

## 🚀 Instalación Rápida con UV

### 1. Instalar UV (si no lo tienes)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Crear entorno virtual e instalar dependencias

```bash
# Crear entorno virtual con UV
uv venv

# Activar entorno virtual
# En PowerShell:
.venv\Scripts\Activate.ps1

# En Linux/macOS:
source .venv/bin/activate

# Instalar dependencias (¡súper rápido!)
uv pip install -e .
```

### 3. Ejecutar el proyecto

```bash
# Navegar a la carpeta del modelo
cd cloro

# Opción 1: Pipeline completo
uv run --no-project ejecutar_pipeline.py

# Opción 2: Paso a paso
uv run --no-project generar_datos_cloro.py
uv run --no-project modelo_cloro_lstm.py
```

**Nota:** Usamos `--no-project` porque son scripts independientes, no un paquete Python instalable.

## 💡 Ventajas de usar UV

- **Velocidad:** 10-100x más rápido que pip
- **Gestión de versiones:** Maneja automáticamente versiones de Python
- **Lock file:** Reproducibilidad garantizada
- **Cache inteligente:** Reutiliza paquetes entre proyectos

## 📦 Comandos Útiles

```bash
# Sincronizar entorno con pyproject.toml
uv sync

# Agregar nueva dependencia
uv add nombre-paquete

# Actualizar todas las dependencias
uv lock --upgrade

# Ejecutar script sin activar entorno
uv run python generar_datos_cloro.py

# Ver dependencias instaladas
uv pip list
```

## 🔄 Migración desde requirements.txt

Si prefieres seguir usando `requirements.txt`, UV también lo soporta:

```bash
uv pip install -r requirements.txt
```

## 🎯 Método Recomendado (uv run)

La forma más simple sin activar el entorno:

```bash
cd cloro

# Generar datos
uv run --no-project generar_datos_cloro.py

# Entrenar modelo
uv run --no-project modelo_cloro_lstm.py
```

UV automáticamente:
1. ✓ Usa el entorno virtual si existe
2. ✓ Mantiene las dependencias instaladas
3. ✓ Ejecuta el script

**Nota:** El flag `--no-project` evita que UV intente instalar el proyecto como paquete.

---

**¡UV hace todo más rápido y simple!** 🚀
