import wntr  # Importa el módulo wntr para trabajar con redes de distribución de agua
import pandas as pd  # Importa el módulo pandas para manejar estructuras de datos
import numpy as np  # Importa el módulo numpy para trabajar con arrays y operaciones matemáticas
import os  # Importa el módulo os para interactuar con el sistema operativo
import matplotlib.pyplot as plt  # Importa el módulo matplotlib.pyplot para realizar gráficos
import sys
sys.path.append("/C:/Users/mmmol/Desktop/USB F INFORMACION ACTUAL 2022/ALGORITMO PSO CON QGIS Y PYTHON 2022/PSO-QPANET-MOD2024- PRUEBAS.py")
# Función para cargar datos desde un archivo .inp de EPANET y ejecutar el modelo en modo estático
def ejecutar_epanet_y_obtener_datos(ruta_archivo_inp):
    try:
        # Cargar el modelo de la red de agua desde el archivo .inp
        red = wntr.network.WaterNetworkModel(ruta_archivo_inp)
        
        # Crear el simulador de EPANET en modo estático
        sim = wntr.sim.EpanetSimulator(red)
        
        # Ejecutar la simulación hidráulica en EPANET en modo estático (un solo paso de tiempo)
        resultados = sim.run_sim()
        
        # Crear un DataFrame para almacenar los datos de las tuberías
        datos_tuberias = []
        
        # Obtener los resultados de presión, caudal y diámetro para cada tubería
        for tuberia_name, tuberia in red.pipes():
            nodo_inicial = tuberia.start_node_name  # Obtener el nodo inicial de la tubería
            nodo_final = tuberia.end_node_name  # Obtener el nodo final de la tubería
            
            # Obtener la presión y caudal para el primer paso de tiempo (índice 0)
            presion = resultados.node['pressure'].loc[0, nodo_inicial]
            caudal = abs(resultados.link['flowrate'].loc[0, tuberia_name])  # Tomar el valor absoluto del caudal
            diametro = red.get_link(tuberia_name).diameter  # Obtener el diámetro de la tubería
            
            # Añadir los datos de la tubería al listado
            datos_tuberias.append({
                'Tubería': tuberia_name,
                'Nodo Inicial': nodo_inicial,
                'Nodo Final': nodo_final,
                'Presión': presion,
                'Caudal': caudal,
                'Diámetro': diametro
            })
        
        # Crear un DataFrame a partir de los datos obtenidos
        df_tuberias = pd.DataFrame(datos_tuberias)
        
        return df_tuberias  # Devolver el DataFrame con los datos de las tuberías
    
    except Exception as e:
        print(f"Error al ejecutar EPANET y obtener datos: {e}")  # Mostrar mensaje de error en caso de excepción
        return None  # Devolver None si hay un error

# Función para calcular la función de penalización y exportar a Excel
def calcular_funcion_penalizacion(df_tuberias, Pmin, Pmax, Cmin, Cmax, Dmin, Dmax):
    try:
        Fun_Pena = []  # Lista para almacenar los valores de la función de penalización
        
        # Iterar sobre cada fila del DataFrame de tuberías
        for index, row in df_tuberias.iterrows():
            presion = row['Presión']
            caudal = abs(row['Caudal'])  # Tomar el valor absoluto del caudal
            diametro = row['Diámetro']
            penal_presion = 10 * (max(0, Pmin - presion, presion - Pmax))**2
            penal_caudal = 100 * (max(0, Cmin - caudal, caudal - Cmax))**2
            penal_diametro = 10 * (max(0, Dmin - diametro, diametro - Dmax))**2
            # Calcular la función de penalización según las condiciones dadas
            if Pmin <= presion <= Pmax and Cmin <= abs(caudal) <= Cmax and Dmin <= diametro <= Dmax:
                penalizacion = abs(presion)  # Penalización basada en la presión
            else:
                #penalizacion = (abs(caudal) * 10000) + (presion**2) + (diametro * 10000)  # Penalización compleja
                penalizacion = penal_presion + penal_caudal + penal_diametro
            
            Fun_Pena.append(penalizacion)  # Añadir el valor de penalización a la lista
        
        df_tuberias['Fun_Penalizacion'] = Fun_Pena  # Añadir la columna de penalización al DataFrame
        
        # Variables y coeficientes del algoritmo PSO
        W = [0.5 for _ in range(len(Fun_Pena))]  # Coeficiente de inercia
        C1 = C2 = [0.5 for _ in range(len(Fun_Pena))]  # Coeficientes cognitivo y social
        Pit = Fun_Pena  # Mejor posición individual
        Pgt = [min(Fun_Pena) for _ in range(len(Fun_Pena))]  # Mejor posición global
        Xit = [min(Pit) for _ in range(len(Fun_Pena))]  # Posición actual
        Vit = [0 for _ in range(len(Fun_Pena))]  # Velocidad
        
        # Gráfica para visualizar la evolución de Xit1
        plt.ion()  # Activar modo interactivo
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        iteracion = 1
        itermax = 100  # Número máximo de iteraciones
        tolerancia = 0.0001  # Tolerancia para detener el algoritmo si los valores no cambian más allá de esta diferencia
        
        # Bucle para iterar hasta alcanzar la convergencia o el máximo de iteraciones
        while iteracion <= itermax:
            iteracion += 1
            inercia = (np.array(W) * np.array(Vit))
            cognitivo = (np.array(C1) * ((np.array(Pit) - np.array(Xit))))
            social = ((np.array(C1) * ((np.array(Pgt) - np.array(Xit)))))
    
            Vit1 = inercia + cognitivo + social  # Actualizar la velocidad
            Xit1 = (Xit + Vit1)  # Actualizar la posición
    
            Pit = Xit1  # Actualizar la mejor posición individual
            
            print('Iteración: ' + str(iteracion) + ' - - - Xit1: ' + str(Xit1))
    
            # Graficar Xit1 en tiempo real
            ax.clear()
            ax.grid(True)
            line1 = ax.plot(Xit1, 'yo')
            ax.set_xlabel('Número de partículas')
            ax.set_ylabel('Valor óptimo')
            ax.set_ylim(min(Xit1) - 0.1, max(Xit1) + 0.1)  # Ajustar límites de y según los valores de Xit1
            plt.pause(0.05)
            
            # Verificar convergencia
            if np.all(np.abs(Xit1 - Xit) < tolerancia):
                print(f"Convergencia alcanzada en la iteración {iteracion}")
                break
            
            Xit = Xit1  # Actualizar la posición actual
        
        # Mostrar todas las partículas convergiendo al valor óptimo
        ax.clear()
        ax.grid(True)
        line1 = ax.plot(Xit1, 'yo')
        ax.set_xlabel('Número de partículas')
        ax.set_ylabel('Valor óptimo')
        ax.set_ylim(min(Xit1) - 0.1, max(Xit1) + 0.1)  # Ajustar límites de y según los valores de Xit1
        plt.pause(1)
        
        # Determinar el nodo óptimo donde se debe instalar la válvula reductora de presión
        indice_min = Fun_Pena.index(min(Fun_Pena))
        valor_optimo = min(Fun_Pena)
        nodo_inicial_optimo = df_tuberias.loc[indice_min, 'Nodo Inicial']
        nodo_final_optimo = df_tuberias.loc[indice_min, 'Nodo Final']
        tuberia_optima = df_tuberias.loc[indice_min, 'Tubería']
        
        print("Nodo inicial óptimo:", nodo_inicial_optimo)
        print("Nodo final óptimo:", nodo_final_optimo)
        print("Tubería óptima:", tuberia_optima)
        
        # Mostrar el valor óptimo en la gráfica
        ax.plot([0, len(Xit1) - 1], [valor_optimo, valor_optimo], 'r--', label='Valor Óptimo')
        ax.legend()
        plt.pause(1)
        
        # Exportar DataFrame a Excel
        script_dir = os.path.dirname(__file__)  # Obtener el directorio del script
        excel_file = os.path.join(script_dir, 'resultados.xlsx')  # Crear la ruta del archivo Excel
        df_tuberias.to_excel(excel_file, index=False)  # Exportar el DataFrame a Excel
        print(f"Datos exportados exitosamente a '{excel_file}'")
        
        return df_tuberias, Fun_Pena  # Devolver el DataFrame y la lista de penalización
    
    except Exception as e:
        print(f"Error al calcular la función de penalización y exportar a Excel: {e}")  # Mostrar mensaje de error en caso de excepción
        return None, None  # Devolver None si hay un error

# Función para seleccionar un archivo .inp mediante GUI (Tkinter)


# Ejecutar el script principal

def pso_discreto_indices(Fun_Pena, n_particulas=None, n_iter=60, w=0.72, c1=1.5, c2=1.5, seed=None, paciencia=12):
    """
    PSO discreto sobre el espacio de índices [0, N-1].
    - Fun_Pena: lista/array de valores de la función objetivo por índice (a minimizar).
    - Retorna: (indice_optimo, valor_optimo).
    """
    import numpy as np
    N = len(Fun_Pena)
    if N <= 0:
        raise ValueError("Fun_Pena vacío; no hay candidatos para evaluar")
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # tamaño del enjambre = N (por defecto)
    if n_particulas is None:
        n_particulas = N

    # Inicialización
    x = rng.uniform(0, N - 1, size=n_particulas)  # posiciones continuas
    v = np.zeros(n_particulas)                    # velocidades
    # pbest
    pbest_x = x.copy()
    pbest_f = np.array([Fun_Pena[int(round(xx))] for xx in x])
    # gbest
    g_idx = int(np.argmin(pbest_f))
    gbest_x = float(pbest_x[g_idx])
    gbest_f = float(pbest_f[g_idx])
    mejor_iter = 0

    for t in range(n_iter):
        r1 = rng.rand(n_particulas)
        r2 = rng.rand(n_particulas)
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        x = x + v
        # Límites y evaluación (snap a índice entero para fitness)
        x = np.clip(x, 0, N - 1)
        for i in range(n_particulas):
            xi = int(round(x[i]))
            f = Fun_Pena[xi]
            # actualizar pbest
            if f < pbest_f[i]:
                pbest_f[i] = f
                pbest_x[i] = x[i]
        # actualizar gbest
        idx_min_local = int(np.argmin(pbest_f))
        if pbest_f[idx_min_local] < gbest_f:
            gbest_f = float(pbest_f[idx_min_local])
            gbest_x = float(pbest_x[idx_min_local])
            mejor_iter = t
        # criterio de parada por paciencia (no-mejora)
        if (t - mejor_iter) >= paciencia:
            break

    indice_optimo = int(round(np.clip(gbest_x, 0, N - 1)))
    valor_optimo = float(gbest_f)
    return indice_optimo, valor_optimo

def proponer_diametro_vrp(caudal_lps, comerciales_pulgadas=(2,3,4,6,8,10), vmin=0.6, vmax=2.5, vtarget=1.5):
    """Calcula un diámetro comercial y la velocidad resultante a partir del caudal (L/s).
    Retorna (pulgadas, velocidad_m_s, diametro_mm)."""
    import math
    Q = float(abs(caudal_lps)) / 1000.0  # L/s -> m^3/s
    pulg_to_mm = {2:50, 3:75, 4:100, 6:150, 8:200, 10:250}
    candidatos = []
    for pin in comerciales_pulgadas:
        dmm = pulg_to_mm.get(pin)
        if not dmm:
            continue
        d = dmm/1000.0
        area = math.pi * (d**2) / 4.0
        if area <= 0:
            continue
        v = Q/area
        if vmin <= v <= vmax:
            score = abs(v - vtarget)
            rank = 0
        else:
            # distancia al rango
            if v < vmin:
                score = vmin - v
            else:
                score = v - vmax
            rank = 1
        candidatos.append((rank, score, abs(v - vtarget), pin, v, dmm))
    if not candidatos:
        return 2, 0.0, 50
    candidatos.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, pin, vsel, dmm = candidatos[0]
    return pin, vsel, dmm


# ==== Helpers añadidos (UI-preservado) ====
import math
import pandas as pd

def _normaliza_col(c):
    import unicodedata
    c = ''.join(ch for ch in unicodedata.normalize('NFKD', str(c)) if not unicodedata.combining(ch))
    return c.lower().strip().replace(' ', '').replace('_','')

def _buscar_col(df, candidatos):
    m = { _normaliza_col(c): c for c in df.columns }
    for cand in candidatos:
        key = _normaliza_col(cand)
        if key in m:
            return m[key]
    for key,c in m.items():
        for cand in candidatos:
            if _normaliza_col(cand) in key:
                return c
    raise KeyError(f"No se encontró ninguna columna entre {candidatos}")

def extraer_diametro_real_mm(df, indice_min):
    col = None
    try:
        col = _buscar_col(df, ['Diámetro','Diametro','Diametro (mm)','D_mm','D(mm)','Diametro_mm','Diameter'])
    except KeyError:
        pass
    if col is None:
        for c in df.columns:
            try:
                vals = pd.to_numeric(df[c], errors='coerce')
                if vals.notna().any():
                    mn, mx = float(vals.min()), float(vals.max())
                    if 20 <= mn <= 1000 and 20 <= mx <= 2000:
                        col = c
                        break
            except Exception:
                pass
    if col is None:
        return 100.0
    val = float(pd.to_numeric(df.loc[indice_min, col], errors='coerce'))
    if 0.5 <= val <= 24:
        return val * 25.4
    return val

def extraer_caudal_lps(df, indice_min):
    cand_lps = ['Caudal','Q','Caudal (L/s)','LPS','lps','Flow (L/s)','Flow_Lps']
    cand_m3s = ['Caudal (m3/s)','m3/s','Q_m3s','Flow (m3/s)','CMS','cms']
    try:
        col = _buscar_col(df, cand_lps + cand_m3s)
    except KeyError:
        col = None
        for c in df.columns:
            try:
                v = float(pd.to_numeric(df.loc[indice_min, c], errors='coerce'))
                if 0.001 <= abs(v) <= 1000:
                    col = c
                    break
            except Exception:
                continue
        if col is None:
            raise KeyError("No se encontró columna de caudal.")
    val = float(pd.to_numeric(df.loc[indice_min, col], errors='coerce'))
    norm = col.lower()
    if 'm3/s' in norm or 'm3s' in norm or 'cms' in norm:
        return val * 1000.0
    return val

def calcular_velocidad_ms(Q_lps, diam_mm):
    Q_m3s = (Q_lps / 1000.0)
    d_m = diam_mm / 1000.0
    import math
    area = math.pi * (d_m**2) / 4.0
    if area <= 0:
        return 0.0
    return Q_m3s / area

def proponer_diametro_vrp(Q_lps, vmin=0.6, vmax=2.5, vtarget=1.5):
    opciones = [(2, 50), (3, 75), (4, 100), (6, 150), (8, 200), (10, 250)]
    cand = []
    for pin, dmm in opciones:
        v = calcular_velocidad_ms(Q_lps, dmm)
        if vmin <= v <= vmax:
            rank, score = 0, 0.0
        else:
            if v < vmin:
                score = vmin - v
            else:
                score = v - vmax
            rank = 1
        cand.append((rank, score, abs(v - vtarget), pin, v, dmm))
    cand.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, pin, vsel, dmm = cand[0]
    return pin, vsel, dmm

def _aprox_pulgadas(mm):
    return round(mm / 25.4)

def mensaje_final(df, indice_min, nodo_optimo):
    Q_lps = extraer_caudal_lps(df, indice_min)
    d_real_mm = extraer_diametro_real_mm(df, indice_min)
    v_real = calcular_velocidad_ms(Q_lps, d_real_mm)
    pin, v_est, d_prop_mm = proponer_diametro_vrp(Q_lps)
    inch_real_aprox = _aprox_pulgadas(d_real_mm)
    msg = (
f"Nodo óptimo {nodo_optimo}.\n"
f"Tubería real (DF): {int(round(d_real_mm))} mm (~{inch_real_aprox}\").\n"
f"Caudal real en tubería: {Q_lps:.2f} L/s.\n"
f"Velocidad real en tubería: {v_real:.2f} m/s.\n"
f"---\n"
f"VRP propuesta: {pin}\" ({int(round(d_prop_mm))} mm), velocidad estimada: {v_est:.2f} m/s."
    )
    return msg
# ==== Fin Helpers ====


# ==== DF/DBF-based extraction improvements ====
import pandas as pd
import numpy as np
import math

_D_COL_CANDIDATES = [
    'Diámetro','Diametro','Diametro (mm)','Diametro_mm','D_mm','D(mm)',
    'Diameter','Diameter (mm)','PipeDiameter','Pipe Diameter',
    'DiametroModelado','Diámetro Modelado','D_model','Diam','D','D[mm]','Diam_mm',
    'Dia','Dia_mm','Tuberia_Diametro','Tubería Diámetro'
]

_Q_COL_CANDIDATES_LPS = [
    'Caudal','Caudal (L/s)','Q','Q (L/s)','LPS','lps','Flow (L/s)','Flow_Lps','Q_lps','Q[L/s]','Q_LPS'
]
_Q_COL_CANDIDATES_M3S = [
    'Caudal (m3/s)','m3/s','Q_m3s','Flow (m3/s)','CMS','cms','Q[m3/s]','Q_M3S'
]

_NODE_COL_CANDIDATES = ['Nodo Inicial','NodoInicial','Nodo','Node','NodeID','NodoID']

def _norm(s):
    import unicodedata
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(ch))
    return s.lower().strip().replace(' ', '').replace('_','')

def _seek_col(df, primary):
    mapping = { _norm(c): c for c in df.columns }
    for cand in primary:
        key = _norm(cand)
        if key in mapping:
            return mapping[key]
    # substring fallback
    for key, orig in mapping.items():
        for cand in primary:
            if _norm(cand) in key:
                return orig
    return None

def _series_float(s):
    return pd.to_numeric(s, errors='coerce')

def _detect_units_mm(series_values):
    """Try to infer if diameter series is in inches or mm using median range."""
    vals = _series_float(series_values).dropna()
    if len(vals) == 0:
        return 'mm'
    med = float(vals.median())
    # Heuristic: if median <= 24, likely inches; if >= 40, likely mm
    if 0.5 <= med <= 24.0:
        return 'in'
    return 'mm'

def extraer_diametro_real_mm(df, indice_min):
    col = _seek_col(df, _D_COL_CANDIDATES)
    if col is None:
        # numeric fallback: pick numeric-like with plausible ranges
        numeric_cols = []
        for c in df.columns:
            vals = _series_float(df[c])
            if vals.notna().sum() > 0:
                numeric_cols.append((c, float(vals.min(skipna=True)), float(vals.max(skipna=True))))
        # prefer something that looks like diameters (20..2000)
        for c, mn, mx in numeric_cols:
            if (20 <= mx <= 2000) or (0.5 <= mx <= 24):
                col = c
                break
        if col is None and numeric_cols:
            col = numeric_cols[0][0]
    val = float(pd.to_numeric(df.loc[indice_min, col], errors='coerce'))
    units = _detect_units_mm(df[col])
    if units == 'in':
        return val * 25.4
    return val

def extraer_caudal_lps(df, indice_min):
    col = _seek_col(df, _Q_COL_CANDIDATES_LPS + _Q_COL_CANDIDATES_M3S)
    if col is None:
        # numeric fallback near flows (0.01..1000)
        for c in df.columns:
            try:
                v = float(pd.to_numeric(df.loc[indice_min, c], errors='coerce'))
                if 0.0001 <= abs(v) <= 5000:
                    col = c
                    break
            except Exception:
                continue
        if col is None:
            raise KeyError("No flow column found.")
    val = float(pd.to_numeric(df.loc[indice_min, col], errors='coerce'))
    norm = _norm(col)
    # Decide units
    if 'm3s' in norm or 'cms' in norm or 'm3/s' in col.lower():
        return val * 1000.0  # m3/s -> L/s
    return val  # assume L/s

def calcular_velocidad_ms(Q_lps, diam_mm):
    Q_m3s = (Q_lps / 1000.0)
    d_m = diam_mm / 1000.0
    area = math.pi * (d_m**2) / 4.0
    if area <= 0:
        return 0.0
    return Q_m3s / area

def proponer_diametro_vrp(Q_lps, vmin=0.6, vmax=2.5, vtarget=1.5):
    """Return (inches, v_est, d_mm) for the best commercial size, using only modeled Q from DF."""
    opciones = [(2, 50), (3, 75), (4, 100), (6, 150), (8, 200), (10, 250)]
    candidates = []
    for pin, dmm in opciones:
        v = calcular_velocidad_ms(Q_lps, dmm)
        if vmin <= v <= vmax:
            rank = 0
            score = 0.0
        else:
            # distance outside the range
            score = (vmin - v) if v < vmin else (v - vmax)
            rank = 1
        candidates.append((rank, score, abs(v - vtarget), pin, v, dmm))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, pin, vsel, dmm = candidates[0]
    return pin, vsel, dmm

def _aprox_in(mm):
    return int(round(mm / 25.4))

def mensaje_final(df, indice_min, nodo_optimo):
    """Builds the exact final ASCII message using DF-sourced Q and D at the chosen node, then proposes commercial VRP."""
    Q_lps = extraer_caudal_lps(df, indice_min)
    d_real_mm = extraer_diametro_real_mm(df, indice_min)
    v_real = calcular_velocidad_ms(Q_lps, d_real_mm)
    pin, v_est, d_prop_mm = proponer_diametro_vrp(Q_lps)
    msg = (
f"Nodo óptimo {nodo_optimo}.\n"
f"Tubería real (DF): {int(round(d_real_mm))} mm (~{_aprox_in(d_real_mm)}\").\n"
f"Caudal real en tubería: {Q_lps:.2f} L/s.\n"
f"Velocidad real en tubería: {v_real:.2f} m/s.\n"
f"---\n"
f"VRP propuesta: {pin}\" ({int(round(d_prop_mm))} mm), velocidad estimada: {v_est:.2f} m/s."
    )
    return msg
# ==== End DF/DBF improvements ====


# ==== Improved non-zero fallback for diameter and flow ====
def _safe_float(x):
    try:
        return float(pd.to_numeric(x, errors='coerce'))
    except Exception:
        return float('nan')

def extraer_diametro_real_mm(df, indice_min):
    # Try multiple candidates and return first plausible (>0) value
    candidates = [
        'Diámetro','Diametro','Diametro (mm)','Diametro_mm','D_mm','D(mm)',
        'Diameter','Diameter (mm)','PipeDiameter','Pipe Diameter',
        'DiametroModelado','Diámetro Modelado','D_model','Diam','D','D[mm]','Diam_mm',
        'Dia','Dia_mm','Tuberia_Diametro','Tubería Diámetro','DIAMETRO','DIAMETRO_MM','DIAMETRO (MM)',
        'Diameter_in','Diametro (in)','DIAMETRO_PULG','D_in'
    ]
    # map of inch-like names
    inch_like = {'Diameter_in','Diametro (in)','DIAMETRO_PULG','D_in'}
    # 1) Named columns
    for col in df.columns:
        name = str(col)
        n = name.lower().replace(' ','').replace('_','')
        for cand in candidates:
            if cand.lower().replace(' ','').replace('_','') in n:
                val = _safe_float(df.loc[indice_min, col])
                if pd.notna(val) and val > 0:
                    # units detection
                    if name in inch_like or (0.5 <= val <= 24):
                        return val * 25.4
                    # if looks like mm but tiny (<=2), treat as inches
                    if val <= 2.5:
                        return val * 25.4
                    return val
    # 2) Numeric heuristic
    for col in df.columns:
        val = _safe_float(df.loc[indice_min, col])
        if pd.notna(val) and val > 0:
            if val <= 24:
                return val * 25.4
            if 20 <= val <= 2000:
                return val
    # fallback
    return 150.0  # assume 6" if nothing else

def extraer_caudal_lps(df, indice_min):
    candidates_lps = ['Caudal','Caudal (L/s)','Q','Q (L/s)','LPS','lps','Flow (L/s)','Flow_Lps','Q_lps','Q[L/s]','Q_LPS','CAUDAL','Q_LPS_MODELADO']
    candidates_m3s = ['Caudal (m3/s)','m3/s','Q_m3s','Flow (m3/s)','CMS','cms','Q[m3/s]','Q_M3S','CAUDAL_M3S','Q_M3S_MODELADO']
    # Named columns first
    for col in df.columns:
        name = str(col).lower().replace(' ','').replace('_','')
        for cand in candidates_lps + candidates_m3s:
            if cand.lower().replace(' ','').replace('_','') in name:
                val = _safe_float(df.loc[indice_min, col])
                if pd.notna(val) and val != 0:
                    if 'm3s' in name or 'cms' in name or 'm3/s' in str(col).lower():
                        return val * 1000.0
                    return val
    # Numeric heuristic near flows
    for col in df.columns:
        val = _safe_float(df.loc[indice_min, col])
        if pd.notna(val) and val != 0:
            if abs(val) < 0.05 and val > 0:  # could be m3/s small
                return val * 1000.0
            if 0.01 <= abs(val) <= 5000:
                return val
    # fallback
    return 10.0
# ==== End improved fallback ====


# ==== Build final message from VRP_DEBUG_DF_SELECTED_ROW.csv ====
def _norm_name(s):
    import unicodedata
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(ch))
    return s.lower().strip().replace(' ','').replace('_','')

def _read_selected_csv(ruta_inp):
    import os, pandas as pd
    csv_path = os.path.join(os.path.dirname(ruta_inp), "VRP_DEBUG_DF_SELECTED_ROW.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df1 = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        # try latin-1 if csv has accents
        df1 = pd.read_csv(csv_path, encoding="latin-1")
    # Some dumps may include headers only; ensure at least one row
    if df1.shape[0] == 0:
        return None
    return df1.iloc[0].to_dict(), list(df1.columns)

def _get_flow_lps_from_row(row):
    # Expect Caudal in m3/s or L/s
    for key in row.keys():
        n = _norm_name(key)
        if 'caudal' in n or 'flow' in n or n == 'q':
            val = float(pd.to_numeric(row[key], errors='coerce'))
            # Decide units by magnitude and header
            if ('m3s' in n) or ('cms' in n) or ('m3/s' in key.lower()) or (val < 0.5):  # likely m3/s
                # If val is tiny (e.g., 0.013), treat as m3/s -> convert
                return val * 1000.0
            # else assume L/s
            return val
    # fallback: 10 L/s
    return 10.0

def _get_diam_mm_from_row(row):
    # Expect diameter in mm or meters
    for key in row.keys():
        n = _norm_name(key)
        if 'diam' in n or 'd' == n:
            val = float(pd.to_numeric(row[key], errors='coerce'))
            # Heuristics:
            #  - if between 0.02 and 2.0 -> meters (e.g., 0.15 m)
            #  - if between 20 and 2000 -> millimeters
            #  - if between 0.5 and 24 -> inches
            if 0.02 <= val <= 2.0:
                return val * 1000.0  # m -> mm
            if 20.0 <= val <= 2000.0:
                return val  # mm
            if 0.5 <= val <= 24.0:
                return val * 25.4  # in -> mm
    # fallback: 150 mm
    return 150.0

def _compute_required_d_mm_by_continuity(Q_lps, v_avg=1.55):
    # A = Q/v ; Q in m3/s, v in m/s -> A in m2
    Q_m3s = Q_lps / 1000.0
    if v_avg <= 0:
        v_avg = 1.55
    A = Q_m3s / v_avg
    if A <= 0:
        return 50.0
    d_m = math.sqrt(4.0 * A / math.pi)
    return d_m * 1000.0  # mm

def _nearest_commercial_inch(d_mm):
    options = [(2, 50), (3, 75), (4, 100), (6, 150), (8, 200), (10, 250)]
    # choose nearest by diameter
    best = min(options, key=lambda t: abs(t[1] - d_mm))
    return best  # (inch, mm)

def mensaje_final_from_selected_csv(ruta_inp, nodo_optimo):
    data = _read_selected_csv(ruta_inp)
    if data is None:
        # Fallback to old method from df
        raise RuntimeError("No se encontró VRP_DEBUG_DF_SELECTED_ROW.csv")
    row, cols = data
    Q_lps = _get_flow_lps_from_row(row)
    d_real_mm = _get_diam_mm_from_row(row)
    # Compute required diameter by continuity at v_avg=(0.6+2.5)/2 = 1.55 m/s
    d_req_mm = _compute_required_d_mm_by_continuity(Q_lps, v_avg=1.55)
    pin, d_prop_mm = _nearest_commercial_inch(d_req_mm)
    v_real = calcular_velocidad_ms(Q_lps, d_real_mm)
    v_est = calcular_velocidad_ms(Q_lps, d_prop_mm)
    msg = (
f"Nodo óptimo {nodo_optimo}.\n"
f"Tubería real (DF): {int(round(d_real_mm))} mm (~{_aprox_in(d_real_mm)}\").\n"
f"Caudal real en tubería: {Q_lps:.2f} L/s.\n"
f"Velocidad real en tubería: {v_real:.2f} m/s.\n"
f"---\n"
f"VRP propuesta: {pin}\" ({int(round(d_prop_mm))} mm), velocidad estimada: {v_est:.2f} m/s."
    )
    return msg
# ==== End CSV-based message ====


# ==== DF-only final message (no debug files needed) ====
def _norm_key(s):
    import unicodedata
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(ch))
    return s.lower().strip().replace(' ','').replace('_','')

def _row_value(row, name_predicates):
    # name_predicates: list of callables taking normalized key -> bool
    for key in row.index:
        nk = _norm_key(str(key))
        if any(pred(nk) for pred in name_predicates):
            try:
                return float(pd.to_numeric(row[key], errors='coerce')), str(key), nk
            except Exception:
                continue
    return float('nan'), None, None

def _get_flow_lps_from_dfrow(row):
    # First pass: look for 'caudal' or 'flow' or 'q'
    val, key, nk = _row_value(row, [
        lambda k: 'caudal' in k,
        lambda k: 'flow' in k,
        lambda k: k == 'q'
    ])
    if pd.isna(val):
        # numeric heuristic as fallback
        for key in row.index:
            try:
                v = float(pd.to_numeric(row[key], errors='coerce'))
            except Exception:
                continue
            if pd.notna(v) and v != 0 and (0.0001 <= abs(v) <= 5000):
                val, nk = v, _norm_key(str(key))
                break
    if pd.isna(val):
        return 10.0
    # Decide units by header + magnitude
    if ('m3' in (nk or '')) or ('cms' in (nk or '')) or ('m3/s' in (key or '').lower()) or (abs(val) <= 2.0):
        # Assume m3/s if magnitude small (<=2) unless clearly L/s
        return val * 1000.0  # to L/s
    return val  # L/s

def _get_diam_mm_from_dfrow(row):
    # Look for 'diam' keys
    val, key, nk = _row_value(row, [lambda k: 'diam' in k or k == 'd'])
    if pd.isna(val):
        # numeric heuristic
        for key in row.index:
            try:
                v = float(pd.to_numeric(row[key], errors='coerce'))
            except Exception:
                continue
            if pd.notna(v) and v > 0:
                if v <= 24.0:
                    return v * 25.4  # inches -> mm
                if 0.02 <= v <= 2.0:
                    return v * 1000.0  # meters -> mm
                if 20.0 <= v <= 2000.0:
                    return v  # mm
    else:
        # interpret units from the found value
        if 0.02 <= val <= 2.0:
            return val * 1000.0  # meters
        if 20.0 <= val <= 2000.0:
            return val  # mm
        if 0.5 <= val <= 24.0:
            return val * 25.4  # inches
    # fallback to 150 mm (6")
    return 150.0

def mensaje_final_from_df(df, indice_min, nodo_optimo, v_avg=1.55):
    # Safe access to row
    row = df.iloc[int(indice_min)]
    Q_lps = _get_flow_lps_from_dfrow(row)
    d_real_mm = _get_diam_mm_from_dfrow(row)
    # Real velocity with actual modeled diameter
    v_real = calcular_velocidad_ms(Q_lps, d_real_mm)
    # Diameter requirement by continuity at v_avg
    d_req_mm = _compute_required_d_mm_by_continuity(Q_lps, v_avg=v_avg)
    pin, d_prop_mm = _nearest_commercial_inch(d_req_mm)
    v_est = calcular_velocidad_ms(Q_lps, d_prop_mm)
    msg = (
f"Nodo óptimo {nodo_optimo}.\n"
f"Tubería real (DF): {int(round(d_real_mm))} mm (~{_aprox_in(d_real_mm)}\").\n"
f"Caudal real en tubería: {Q_lps:.2f} L/s.\n"
f"Velocidad real en tubería: {v_real:.2f} m/s.\n"
f"---\n"
f"VRP propuesta: {pin}\" ({int(round(d_prop_mm))} mm), velocidad estimada: {v_est:.2f} m/s."
    )
    return msg
# ==== End DF-only message ====
