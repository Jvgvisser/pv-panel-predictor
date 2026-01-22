from pydantic import BaseModel, Field

class PanelConfig(BaseModel):
    # Identification
    panel_id: str = Field(..., description="Unique ID for this panel (e.g., p01)")
    friendly_name: str = Field("New Panel", description="Human readable name for the dashboard")
    entity_id: str = Field(..., description="Home Assistant entity_id (lifetime energy sensor)")

    # Solar Geometry
    tilt_deg: float = Field(35.0, ge=0.0, le=90.0, description="Tilt: 0=flat, 90=vertical")
    azimuth_deg: float = Field(
        0.0,
        ge=-180.0,
        le=180.0,
        description="Orientation: 0=south, -90=east, 90=west, +/-180=north",
    )

    # Location (defaults to Netherlands center if not provided)
    latitude: float = Field(52.1326, ge=-90.0, le=90.0)
    longitude: float = Field(5.2913, ge=-180.0, le=180.0)

    # Home Assistant Connection
    ha_base_url: str = Field("", description="e.g. http://192.168.1.xxx:8123")
    ha_token: str = Field("", description="Home Assistant long-lived access token")

    # Technical Specs
    watt_peak: float = Field(400.0, gt=0.0, description="Watt-peak capacity of the individual panel/string")

    # Unit Conversion
    # If HA sensor is in Wh, use 0.001 to get kWh. If already in kWh, use 1.0.
    scale_to_kwh: float = Field(0.001, gt=0.0, description="Multiplier to convert sensor units to kWh")

    class Config:
        # Dit zorgt ervoor dat we extra velden in de JSON negeren in plaats van dat de API faalt
        extra = "ignore"
        schema_extra = {
            "example": {
                "panel_id": "p01",
                "friendly_name": "South Roof Main",
                "entity_id": "sensor.solar_total_energy",
                "tilt_deg": 35.0,
                "azimuth_deg": 0.0,
                "latitude": 52.3676,
                "longitude": 4.9041,
                "ha_base_url": "http://192.168.1.100:8123",
                "ha_token": "your_token_here",
                "watt_peak": 410.0,
                "scale_to_kwh": 0.001
            }
        }