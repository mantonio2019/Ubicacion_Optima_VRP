# -*- coding: utf-8 -*-
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
import os

from .Ubicacion_Optima_VRP_dialog import UbicacionOptimaVrpDialogBase
from . import pso_optimizador

class UbicacionOptimaVrp:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dlg = None

    def tr(self, message):
        return QCoreApplication.translate('UbicacionOptimaVrp', message)

    def initGui(self):
        self.action = QAction(QIcon(os.path.join(self.plugin_dir, 'icon.png')), self.tr('Ubicacion Optima VRP'), self.iface.mainWindow())
        self.action.triggered.connect(self.open_dialog)
        self.iface.addPluginToMenu(self.tr('&Ubicacion Optima VRP'), self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.action:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginMenu(self.tr('&Ubicacion Optima VRP'), self.action)
            self.action = None

    def open_dialog(self):
        self.dlg = UbicacionOptimaVrpDialogBase()
        # Connect UI buttons if present
        if hasattr(self.dlg, 'pushButtonBrowse'):
            self.dlg.pushButtonBrowse.clicked.connect(self.on_browse_inp)
        if hasattr(self.dlg, 'pushButtonRun'):
            self.dlg.pushButtonRun.clicked.connect(self.on_run_clicked)
        self.dlg.show()

    def on_browse_inp(self):
        ruta, _ = QFileDialog.getOpenFileName(self.iface.mainWindow(), self.tr('Selecciona archivo INP'), '', self.tr('EPANET (*.inp)'))
        if ruta and hasattr(self.dlg, 'lineEditINP'):
            self.dlg.lineEditINP.setText(ruta)

    def _read_params(self):
        # Helper to read UI parameters safely
        ruta_inp = self.dlg.lineEditINP.text().strip() if hasattr(self.dlg, 'lineEditINP') else ''
        if not ruta_inp:
            raise ValueError('Selecciona un archivo INP.')
        def _read_float(obj_name, default):
            try:
                if hasattr(self.dlg, obj_name):
                    return float(getattr(self.dlg, obj_name).text())
            except Exception:
                pass
            return default
        Pmin = _read_float('lineEditPmin', 30.0)
        Pmax = _read_float('lineEditPmax', 35.0)
        Cmin = _read_float('lineEditCmin', 0.5)
        Cmax = _read_float('lineEditCmax', 5.0)
        Dmin = _read_float('lineEditDmin', 50.0)
        Dmax = _read_float('lineEditDmax', 250.0)
        return ruta_inp, Pmin, Pmax, Cmin, Cmax, Dmin, Dmax

    def _select_index_from_penal(self, penal):
        # Robust selection of best index even if PSO symbols are missing
        try:
            if hasattr(pso_optimizador, 'pso_discreto_indice_penal'):
                return pso_optimizador.pso_discreto_indice_penal(penal, n_iter=60, w=0.72, c1=1.5, c2=1.5, seed=None, paciencia=12)
            if hasattr(pso_optimizador, 'pso_discreto_indice'):
                return pso_optimizador.pso_discreto_indice(penal, n_iter=60, w=0.72, c1=1.5, c2=1.5, seed=None, paciencia=12)
        except Exception:
            # fall through to minimal selection
            pass
        try:
            import numpy as _np
            try:
                import pandas as _pd
            except Exception:
                _pd = None
            if _pd is not None and hasattr(penal, 'values'):
                try:
                    vals = _pd.to_numeric(_pd.Series(getattr(penal, 'values', penal)), errors='coerce').astype(float).values
                except Exception:
                    vals = _np.asarray(penal).astype(float).ravel()
            else:
                vals = _np.asarray(penal).astype(float).ravel()
            idx = int(_np.nanargmin(vals))
            return idx, float(vals[idx])
        except Exception:
            return 0, 0.0

    def on_run_clicked(self):
        try:
            ruta_inp, Pmin, Pmax, Cmin, Cmax, Dmin, Dmax = self._read_params()
            df = pso_optimizador.ejecutar_epanet_y_obtener_datos(ruta_inp)
            df, penal = pso_optimizador.calcular_funcion_penalizacion(df, Pmin, Pmax, Cmin, Cmax, Dmin, Dmax)
            indice_min, _valor = self._select_index_from_penal(penal)
            
            # Nodo optimo (col tolerant)
            col_candidates = ['Nodo Inicial','NodoInicial','Nodo','Node','NodeID','NodoID']
            def _norm(s):
                import unicodedata
                s = ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(ch))
                return s.lower().strip().replace(' ','').replace('_','')
            mapping = { _norm(c): c for c in df.columns }
            col_nodo = None
            for cand in col_candidates:
                key = _norm(cand)
                if key in mapping:
                    col_nodo = mapping[key]
                    break
            if col_nodo is None:
                # fallback first column
                col_nodo = df.columns[0]
            nodo_optimo = str(df.loc[indice_min, col_nodo])
            # Final message
            msg = pso_optimizador.mensaje_final_from_df(df, indice_min, nodo_optimo)
            QMessageBox.information(None, "Resultado", msg)
        except Exception as e:
            QMessageBox.critical(None, "Error", str(e))
