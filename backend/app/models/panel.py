from pydantic import BaseModel, Field

class PanelConfig(BaseModel):
    panel_id: str = Field(..., description="Unique id for this panel/model")
    friendly_name: str = Field("", description="Human readable name for the panel") # Nieuw
    entity_id: str = Field(..., description="Home Assistant entity_id (lifetime energy sensor)")

    tilt_deg: float = Field(35.0, ge=0.0, le=90.0, description="0=flat, 90=vertical")
    azimuth_deg: float = Field(
        0.0,
        ge=-180.0,
        le=180.0,
        description="Open-Meteo: 0=south, -90=east, 90=west, +/-180=north",
    )

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

    ha_base_url: str = Field(..., description="e.g. http://homeassistant.local:8123")
    ha_token: str = Field(..., description="Home Assistant long-lived access token")

    # Nieuw: Watt-peak van het paneel
    watt_peak: float = Field(400.0, gt=0.0, description="Watt-peak capacity of the panel")

    # Convert lifetime units to kWh. Wh -> 0.001, kWh -> 1.0
    scale_to_kwh: float = Field(0.001, gt=0.0, description="Multiply lifetime-diff by this to get kWh")