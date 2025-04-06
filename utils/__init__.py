import sys
import subprocess

def ensure_dependencies():
    """Asegura que todas las dependencias estén instaladas"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            "requirements.txt", "--no-cache-dir"
        ])
        return True
    except Exception as e:
        print(f"Error instalando dependencias: {e}")
        return False
```
# This file marks the utils directory as a Python package
