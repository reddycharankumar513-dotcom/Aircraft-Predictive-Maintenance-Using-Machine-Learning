"""
Configuration and domain metadata for the Aircraft Predictive Maintenance System.
Aligned with NASA C-MAPSS turbofan engine degradation benchmarks and Airbus Skywise principles.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CMAPSS_DIR = PROJECT_ROOT / "CMAPSSData"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# NASA C-MAPSS Column Definitions
INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

TARGET_COL = "RUL"

# Default Piecewise Linear RUL Clip Threshold
# In turbofan prognostics, engines show minimal measurable degradation during early cycles.
# Standard PHM08 benchmark literature caps RUL at 125 cycles.
DEFAULT_RUL_CLIP = 125

# Aerospace Maintenance Risk Tiers (Cycles Remaining)
RISK_THRESHOLDS = {
    "CRITICAL": 30,    # Immediate maintenance: High risk of Aircraft On Ground (AOG)
    "HIGH": 75,        # Next scheduled turnaround: Pre-order Line Replaceable Units (LRU)
    "MEDIUM": 125,     # Increased monitoring: Watch for degradation trend acceleration
    "LOW": float("inf") # Normal flight operations
}

RISK_COLORS = {
    "CRITICAL": "#DC2626", # Red
    "HIGH": "#EA580C",     # Orange
    "MEDIUM": "#CA8A04",   # Amber/Yellow
    "LOW": "#16A34A"       # Green
}

# Detailed Engineering Metadata for C-MAPSS Sensors
SENSOR_METADATA = {
    "s_1": {"name": "T2", "desc": "Total temperature at fan inlet", "unit": "°R", "subsystem": "Inlet/Fan"},
    "s_2": {"name": "T24", "desc": "Total temperature at LPC outlet", "unit": "°R", "subsystem": "Low Pressure Compressor"},
    "s_3": {"name": "T30", "desc": "Total temperature at HPC outlet", "unit": "°R", "subsystem": "High Pressure Compressor"},
    "s_4": {"name": "T50", "desc": "Total temperature at LPT outlet", "unit": "°R", "subsystem": "Low Pressure Turbine"},
    "s_5": {"name": "P2", "desc": "Pressure at fan inlet", "unit": "psia", "subsystem": "Inlet/Fan"},
    "s_6": {"name": "P15", "desc": "Total pressure in bypass-duct", "unit": "psia", "subsystem": "Bypass Duct"},
    "s_7": {"name": "P30", "desc": "Total pressure at HPC outlet", "unit": "psia", "subsystem": "High Pressure Compressor"},
    "s_8": {"name": "Nf", "desc": "Physical fan speed", "unit": "rpm", "subsystem": "Fan Rotor"},
    "s_9": {"name": "Nc", "desc": "Physical core speed", "unit": "rpm", "subsystem": "Core Spool"},
    "s_10": {"name": "epr", "desc": "Engine pressure ratio (P50/P2)", "unit": "-", "subsystem": "Thermodynamics"},
    "s_11": {"name": "Ps30", "desc": "Static pressure at HPC outlet", "unit": "psia", "subsystem": "High Pressure Compressor"},
    "s_12": {"name": "phi", "desc": "Ratio of fuel flow to Ps30", "unit": "pps/psia", "subsystem": "Fuel System"},
    "s_13": {"name": "NRf", "desc": "Corrected fan speed", "unit": "rpm", "subsystem": "Fan Rotor"},
    "s_14": {"name": "NRc", "desc": "Corrected core speed", "unit": "rpm", "subsystem": "Core Spool"},
    "s_15": {"name": "BPR", "desc": "Bypass Ratio", "unit": "-", "subsystem": "Flow Split"},
    "s_16": {"name": "farB", "desc": "Burner fuel-air ratio", "unit": "-", "subsystem": "Combustor"},
    "s_17": {"name": "htBleed", "desc": "Bleed Enthalpy", "unit": "-", "subsystem": "Bleed Air System"},
    "s_18": {"name": "Nf_dmd", "desc": "Demanded fan speed", "unit": "rpm", "subsystem": "FADEC Control"},
    "s_19": {"name": "PCNfR_dmd", "desc": "Demanded corrected fan speed", "unit": "rpm", "subsystem": "FADEC Control"},
    "s_20": {"name": "W31", "desc": "HPT coolant bleed", "unit": "lbm/s", "subsystem": "High Pressure Turbine"},
    "s_21": {"name": "W32", "desc": "LPT coolant bleed", "unit": "lbm/s", "subsystem": "Low Pressure Turbine"}
}

# Sensors with non-informative / zero variance under sea-level single operating conditions (FD001)
FD001_CONSTANT_SENSORS = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19", "setting_3"]

# Informative subset of sensors commonly tracked for degradation in FD001
PRIMARY_SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
]
