"""Configuration for Dispatch IQ backend."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys — read from environment / .env only; never hardcode secrets in source.
ERCOT_API_KEY = os.getenv("ERCOT_API_KEY", "")
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# BTM Generator Economics
HEAT_RATE = 7.5          # MMBtu/MWh — typical gas turbine
O_AND_M_COST = 3.50      # $/MWh — variable O&M
FACILITY_SIZE_MW = 200   # MW — data center facility size

# Gas Pricing Defaults (Waha Hub)
DEFAULT_GAS_PRICE = 2.41  # $/MMBtu — Waha spot
HENRY_HUB_PREMIUM = 0.80  # $/MMBtu — typical HH premium over Waha

# ERCOT Settlement Points for candidate sites
SETTLEMENT_POINTS = {
    "midland": "LZ_WEST",
    "odessa": "LZ_WEST",
    "abilene": "LZ_WEST",
    "houston": "LZ_HOUSTON",
    "dallas": "LZ_NORTH",
    "san_antonio": "LZ_SOUTH",
    "tucson": "PALO_VERDE",  # WECC node
}
