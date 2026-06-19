import sys
import os

print("="*60)
print(" AVISO IMPORTANTE: CAMBIO ARQUITECTÓNICO")
print("="*60)
print("De acuerdo con la remediación de Data Leakage y la ")
print("implementación del estándar institucional Walk-Forward, ")
print("la inferencia ahora se realiza de forma combinada durante ")
print("el entrenamiento en el script 'finetune_tspulse.py'.")
print("")
print("Por favor, ejecuta 'python models/finetune_tspulse.py' para ")
print("entrenar y generar las señales en 'data/tspulse_signals.json'.")
print("Este archivo precalculador ya no es necesario y ha sido desactivado")
print("para prevenir la sobrescritura accidental de señales OOS.")
print("="*60)
sys.exit(0)
