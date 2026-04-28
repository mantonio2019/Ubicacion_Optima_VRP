"""
PSO_QPANET.py
Capa de compatibilidad: delega funciones al módulo pso_optimizador
para evitar duplicación de lógica y mantener consistencia con el plugin.
"""

from . import pso_optimizador

def ejecutar_epanet_y_obtener_datos(ruta_archivo_inp):
    return pso_optimizador.ejecutar_epanet_y_obtener_datos(ruta_archivo_inp)

def calcular_funcion_penalizacion(df_tuberias, Pmin, Pmax, Cmin, Cmax, Dmin, Dmax):
    return pso_optimizador.calcular_funcion_penalizacion(df_tuberias, Pmin, Pmax, Cmin, Cmax, Dmin, Dmax)
