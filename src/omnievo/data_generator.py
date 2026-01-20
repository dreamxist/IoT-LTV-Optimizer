"""
Generador de datos sinteticos - Smart Music Festival
basado en el paper de Pramono et al pero con algunas modificaciones
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# config de canales, sacado del paper
CHANNEL_CONFIG = {
    # digital
    "facebook_ads": {"type": "digital", "range": (0, 15)},
    "email_opens": {"type": "digital", "range": (0, 10)},
    "web_visits": {"type": "digital", "range": (0, 20)},
    # app
    "app_opens": {"type": "app", "range": (0, 30)},
    "ticket_purchase": {"type": "app", "range": (0, 3)},
    "wallet_topup": {"type": "app", "range": (0, 5)},
    # iot/fisico
    "rfid_entry": {"type": "iot", "range": (0, 5)},
    "stage_dwell_time": {"type": "iot", "range": (0, 120)},  # mins
    "nfc_purchases": {"type": "iot", "range": (0, 20)},
}


# perfiles de usuario - esto lo ajuste varias veces hasta que quedo bien
SEGMENT_PROFILES = {
    "VIP": {
        "prob": 0.15,
        "ltv": (80, 150),
        "weights": {
            "facebook_ads": 0.8, "email_opens": 0.9, "web_visits": 0.7,
            "app_opens": 0.95, "ticket_purchase": 1.0, "wallet_topup": 0.9,
            "rfid_entry": 1.0, "stage_dwell_time": 0.9, "nfc_purchases": 0.95,
        },
    },
    "Regular": {
        "prob": 0.55,
        "ltv": (30, 80),
        "weights": {
            "facebook_ads": 0.5, "email_opens": 0.4, "web_visits": 0.5,
            "app_opens": 0.6, "ticket_purchase": 0.8, "wallet_topup": 0.5,
            "rfid_entry": 0.7, "stage_dwell_time": 0.6, "nfc_purchases": 0.5,
        },
    },
    "Digital-Only": {
        "prob": 0.20,
        "ltv": (5, 30),
        "weights": {
            "facebook_ads": 0.7, "email_opens": 0.5, "web_visits": 0.8,
            "app_opens": 0.4, "ticket_purchase": 0.3, "wallet_topup": 0.1,
            "rfid_entry": 0.0, "stage_dwell_time": 0.0, "nfc_purchases": 0.0,
        },
    },
    "Casual": {
        "prob": 0.10,
        "ltv": (0, 20),
        "weights": {
            "facebook_ads": 0.3, "email_opens": 0.1, "web_visits": 0.2,
            "app_opens": 0.2, "ticket_purchase": 0.1, "wallet_topup": 0.0,
            "rfid_entry": 0.1, "stage_dwell_time": 0.2, "nfc_purchases": 0.1,
        },
    },
}


class DataGenerator:
    """genera datos sinteticos para el festival"""

    def __init__(self, n_users=1000, channels=None, segments=None,
                 noise_level=0.2, random_state=42):
        self.n_users = n_users
        self.channels = channels or CHANNEL_CONFIG
        self.segments = segments or SEGMENT_PROFILES
        self.noise_level = noise_level
        self.seed = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # validar que las probs sumen 1
        total = sum(s["prob"] for s in self.segments.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"probs deben sumar 1, tienes {total}")

        self._scaler = None  # se llena despues

    def generate(self, normalize=True):
        """genera el dataset"""
        # asignar segmentos
        seg_names = list(self.segments.keys())
        seg_probs = [self.segments[s]["prob"] for s in seg_names]
        user_segs = np.random.choice(seg_names, size=self.n_users, p=seg_probs)

        data = []
        for i, seg_name in enumerate(user_segs):
            seg = self.segments[seg_name]
            row = self._gen_user(i, seg_name, seg)
            data.append(row)

        df = pd.DataFrame(data)

        if normalize:
            cols = list(self.channels.keys())
            self._scaler = MinMaxScaler()
            df[cols] = self._scaler.fit_transform(df[cols])

        return df

    def _gen_user(self, uid, seg_name, seg):
        """genera un usuario"""
        row = {"user_id": f"u{uid}", "segment": seg_name}

        weights = seg["weights"]
        for ch, cfg in self.channels.items():
            w = weights.get(ch, 0.5)
            lo, hi = cfg["range"]

            # valor base + ruido
            base = w * hi
            noise = np.random.normal(0, self.noise_level * hi)
            val = np.clip(base + noise, lo, hi)

            # redondear si es discreto
            if hi <= 30:
                val = int(round(val))
            row[ch] = val

        # calcular LTV
        ltv_lo, ltv_hi = seg["ltv"]
        base_ltv = np.random.uniform(ltv_lo, ltv_hi)

        # bonus por interacciones fisicas (esto correlaciona mejor con LTV real)
        iot_bonus = (
            row.get("rfid_entry", 0) * 2 +
            row.get("nfc_purchases", 0) * 1.5 +
            row.get("stage_dwell_time", 0) * 0.1
        )
        row["LTV_real"] = base_ltv + iot_bonus * self.noise_level

        return row

    def get_channel_names(self):
        return list(self.channels.keys())

    def get_channel_types(self):
        """agrupa canales por tipo"""
        types = {}
        for ch, cfg in self.channels.items():
            t = cfg["type"]
            if t not in types:
                types[t] = []
            types[t].append(ch)
        return types


def generate_simple_dataset(n_users=200, channels=None, random_state=42):
    """
    version simplificada para pruebas rapidas
    compatible con el notebook original
    """
    if channels is None:
        channels = ["facebook_ads", "app_opens", "rfid_entry", "nfc_purchases"]

    # config reducida
    cfg = {ch: CHANNEL_CONFIG.get(ch, {"type": "other", "range": (0, 20)})
           for ch in channels}

    # segmentos simplificados
    segs = {
        "VIP": {
            "prob": 0.2, "ltv": (50, 100),
            "weights": {ch: 0.9 for ch in channels},
        },
        "Regular": {
            "prob": 0.6, "ltv": (20, 50),
            "weights": {ch: 0.5 for ch in channels},
        },
        "Digital-Only": {
            "prob": 0.2, "ltv": (0, 20),
            "weights": {"facebook_ads": 0.6, "app_opens": 0.4,
                       "rfid_entry": 0.0, "nfc_purchases": 0.0},
        },
    }

    gen = DataGenerator(n_users=n_users, channels=cfg, segments=segs,
                       random_state=random_state)
    return gen.generate(normalize=True)


