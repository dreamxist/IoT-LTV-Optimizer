"""
Generador de Datos Sintéticos para el Smart Music Festival.

Este módulo simula el Customer Journey completo integrando tres capas de datos:
- Capa Digital (AdTech): Interacciones de marketing
- Capa Transaccional (App): Compras y uso de la app
- Capa Física (IoT): Sensores RFID/NFC en el evento
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Configuración de canales por defecto
CHANNEL_CONFIG = {
    # Capa Digital
    "facebook_ads": {"type": "digital", "range": (0, 15)},
    "email_opens": {"type": "digital", "range": (0, 10)},
    "web_visits": {"type": "digital", "range": (0, 20)},
    # Capa App/Transaccional
    "app_opens": {"type": "app", "range": (0, 30)},
    "ticket_purchase": {"type": "app", "range": (0, 3)},
    "wallet_topup": {"type": "app", "range": (0, 5)},
    # Capa Física (IoT)
    "rfid_entry": {"type": "iot", "range": (0, 5)},
    "stage_dwell_time": {"type": "iot", "range": (0, 120)},  # minutos
    "nfc_purchases": {"type": "iot", "range": (0, 20)},
}


@dataclass
class SegmentProfile:
    """Perfil de comportamiento por segmento de usuario."""

    name: str
    probability: float
    ltv_range: tuple[float, float]
    channel_weights: dict[str, float] = field(default_factory=dict)


# Perfiles de segmentos por defecto
DEFAULT_SEGMENTS = [
    SegmentProfile(
        name="VIP",
        probability=0.15,
        ltv_range=(80, 150),
        channel_weights={
            "facebook_ads": 0.8,
            "email_opens": 0.9,
            "web_visits": 0.7,
            "app_opens": 0.95,
            "ticket_purchase": 1.0,
            "wallet_topup": 0.9,
            "rfid_entry": 1.0,
            "stage_dwell_time": 0.9,
            "nfc_purchases": 0.95,
        },
    ),
    SegmentProfile(
        name="Regular",
        probability=0.55,
        ltv_range=(30, 80),
        channel_weights={
            "facebook_ads": 0.5,
            "email_opens": 0.4,
            "web_visits": 0.5,
            "app_opens": 0.6,
            "ticket_purchase": 0.8,
            "wallet_topup": 0.5,
            "rfid_entry": 0.7,
            "stage_dwell_time": 0.6,
            "nfc_purchases": 0.5,
        },
    ),
    SegmentProfile(
        name="Digital-Only",
        probability=0.20,
        ltv_range=(5, 30),
        channel_weights={
            "facebook_ads": 0.7,
            "email_opens": 0.5,
            "web_visits": 0.8,
            "app_opens": 0.4,
            "ticket_purchase": 0.3,
            "wallet_topup": 0.1,
            "rfid_entry": 0.0,  # No asiste físicamente
            "stage_dwell_time": 0.0,
            "nfc_purchases": 0.0,
        },
    ),
    SegmentProfile(
        name="Casual",
        probability=0.10,
        ltv_range=(0, 20),
        channel_weights={
            "facebook_ads": 0.3,
            "email_opens": 0.1,
            "web_visits": 0.2,
            "app_opens": 0.2,
            "ticket_purchase": 0.1,
            "wallet_topup": 0.0,
            "rfid_entry": 0.1,
            "stage_dwell_time": 0.2,
            "nfc_purchases": 0.1,
        },
    ),
]


class DataGenerator:
    """
    Generador de datos sintéticos para simulación de Smart Music Festival.

    Genera usuarios con comportamientos diferenciados por segmento,
    simulando interacciones en canales digitales, app e IoT.

    Attributes:
        n_users: Número de usuarios a generar
        channels: Configuración de canales
        segments: Perfiles de segmentos
        noise_level: Nivel de ruido aleatorio (0-1)
        random_state: Semilla para reproducibilidad
    """

    def __init__(
        self,
        n_users: int = 1000,
        channels: Optional[dict] = None,
        segments: Optional[list[SegmentProfile]] = None,
        noise_level: float = 0.2,
        random_state: Optional[int] = 42,
    ):
        self.n_users = n_users
        self.channels = channels or CHANNEL_CONFIG
        self.segments = segments or DEFAULT_SEGMENTS
        self.noise_level = noise_level
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Validar probabilidades de segmentos
        total_prob = sum(s.probability for s in self.segments)
        if not np.isclose(total_prob, 1.0):
            raise ValueError(f"Las probabilidades de segmentos deben sumar 1.0, got {total_prob}")

    def generate(self, normalize: bool = True) -> pd.DataFrame:
        """
        Genera el dataset sintético completo.

        Args:
            normalize: Si True, aplica Min-Max scaling a los canales

        Returns:
            DataFrame con columnas: user_id, segment, [canales...], LTV_real
        """
        # Asignar segmentos a usuarios
        segment_names = [s.name for s in self.segments]
        segment_probs = [s.probability for s in self.segments]
        user_segments = np.random.choice(segment_names, size=self.n_users, p=segment_probs)

        # Crear diccionario de segmentos para acceso rápido
        segment_dict = {s.name: s for s in self.segments}

        # Generar datos por usuario
        data = []
        for i, segment_name in enumerate(user_segments):
            segment = segment_dict[segment_name]
            row = self._generate_user(i, segment)
            data.append(row)

        df = pd.DataFrame(data)

        # Normalizar canales si se solicita
        if normalize:
            channel_cols = list(self.channels.keys())
            scaler = MinMaxScaler()
            df[channel_cols] = scaler.fit_transform(df[channel_cols])
            self._scaler = scaler

        return df

    def _generate_user(self, user_id: int, segment: SegmentProfile) -> dict:
        """Genera los datos de un usuario individual."""
        row = {
            "user_id": f"user_{user_id}",
            "segment": segment.name,
        }

        # Generar valor para cada canal
        for channel, config in self.channels.items():
            weight = segment.channel_weights.get(channel, 0.5)
            min_val, max_val = config["range"]

            # Valor base según peso del segmento
            base_value = weight * max_val

            # Añadir ruido
            noise = np.random.normal(0, self.noise_level * max_val)
            value = base_value + noise

            # Clamp al rango válido
            value = np.clip(value, min_val, max_val)

            # Convertir a entero si el rango es discreto
            if max_val <= 30:
                value = int(round(value))

            row[channel] = value

        # Generar LTV real basado en el segmento
        ltv_min, ltv_max = segment.ltv_range
        base_ltv = np.random.uniform(ltv_min, ltv_max)

        # El LTV también depende de las interacciones IoT (correlación realista)
        iot_bonus = (
            row.get("rfid_entry", 0) * 2 +
            row.get("nfc_purchases", 0) * 1.5 +
            row.get("stage_dwell_time", 0) * 0.1
        )

        row["LTV_real"] = base_ltv + iot_bonus * self.noise_level

        return row

    def get_channel_names(self) -> list[str]:
        """Retorna la lista de nombres de canales."""
        return list(self.channels.keys())

    def get_channel_types(self) -> dict[str, list[str]]:
        """Retorna los canales agrupados por tipo."""
        types = {}
        for channel, config in self.channels.items():
            channel_type = config["type"]
            if channel_type not in types:
                types[channel_type] = []
            types[channel_type].append(channel)
        return types


def generate_simple_dataset(
    n_users: int = 200,
    channels: list[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Función de conveniencia para generar un dataset simple (4 canales).

    Compatible con el notebook original para mantener retrocompatibilidad.

    Args:
        n_users: Número de usuarios
        channels: Lista de canales (default: facebook_ads, app_opens, rfid_entry, nfc_purchases)
        random_state: Semilla aleatoria

    Returns:
        DataFrame normalizado con los canales especificados
    """
    if channels is None:
        channels = ["facebook_ads", "app_opens", "rfid_entry", "nfc_purchases"]

    # Configuración simplificada
    simple_config = {ch: CHANNEL_CONFIG.get(ch, {"type": "other", "range": (0, 20)})
                     for ch in channels}

    # Segmentos simplificados
    simple_segments = [
        SegmentProfile(
            name="VIP",
            probability=0.2,
            ltv_range=(50, 100),
            channel_weights={ch: 0.9 for ch in channels},
        ),
        SegmentProfile(
            name="Regular",
            probability=0.6,
            ltv_range=(20, 50),
            channel_weights={ch: 0.5 for ch in channels},
        ),
        SegmentProfile(
            name="Digital-Only",
            probability=0.2,
            ltv_range=(0, 20),
            channel_weights={
                "facebook_ads": 0.6,
                "app_opens": 0.4,
                "rfid_entry": 0.0,
                "nfc_purchases": 0.0,
            },
        ),
    ]

    generator = DataGenerator(
        n_users=n_users,
        channels=simple_config,
        segments=simple_segments,
        random_state=random_state,
    )

    return generator.generate(normalize=True)
