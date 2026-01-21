from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    out = df.copy()

    # Support both: time as column OR as DatetimeIndex
    if time_col in out.columns:
        t = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        t = out.index
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
    else:
        raise ValueError("add_time_features: need 'time' column or DatetimeIndex")

    out["hour"] = (t.dt.hour if hasattr(t, "dt") else t.hour)
    out["doy"] = (t.dt.dayofyear if hasattr(t, "dt") else t.dayofyear)out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def _ensure_dtindex(obj, *, kind: str) -> pd.DataFrame:
    """
    Make sure we have a DataFrame with a tz-aware UTC DatetimeIndex.
    Accepts:
      - pd.Series (becomes DataFrame)
      - pd.DataFrame with DatetimeIndex
      - pd.DataFrame with a time-like column: time/start/datetime/timestamp/date/last_changed
    """
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=kind)
    else:
        df = obj.copy()

    # Already good?
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")
        return df

    # Try find a usable time column
    candidates = ["time", "ts", "start", "end", "datetime", "timestamp", "date", "last_changed", "last_updated"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        # Sometimes it's nested under index name
        if df.index.name in candidates:
            df = df.reset_index()
            col = df.columns[0]  # the reset index column
        else:
            raise ValueError(f"{kind} must have DatetimeIndex or one of columns {candidates}. Got columns={list(df.columns)}")

    t = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.assign(_time=t).dropna(subset=["_time"]).drop(columns=[c for c in [col] if c in df.columns])
    df = df.set_index("_time").sort_index()

    # Remove duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_training_frame(energy: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    # Always use DST-safe index-join variant
    return build_training_frame_idxjoin(energy, meteo)


@dataclass
class TrainedModel:
    model: LGBMRegressor
    features: list


class PanelModelService:
    def __init__(self, base_dir: str = "data/models") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _panel_dir(self, panel_id: str) -> Path:
        d = self.base_dir / panel_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def train(self, panel_id: str, df: pd.DataFrame) -> dict:
        features = [
            "global_tilted_irradiance",
            "shortwave_radiation",
            "cloud_cover",
            "temperature_2m",
            "kwh_lag_24",
            "gti_lag_24",
            "hour_sin", "hour_cos",
            "doy_sin", "doy_cos",
        ]

        df = df.sort_values("time") if "time" in df.columns else df.sort_index()

        # Validation: last 7 days (168h) when available
        val_n = min(168, max(24, int(len(df) * 0.15)))
        train_df = df.iloc[:-val_n] if len(df) > val_n else df
        val_df = df.iloc[-val_n:] if len(df) > val_n else df.iloc[0:0]

        X_train = train_df[features]
        y_train = train_df["kwh"]

        model = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=-1,
        )

        metrics = {"train_rows": int(len(train_df))}
        if len(val_df) > 0:
            X_val = val_df[features]
            y_val = val_df["kwh"]

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l1",
                callbacks=[],
            )

            pred = np.clip(model.predict(X_val), 0, None)
            metrics["val_rows"] = int(len(val_df))
            metrics["val_mae_kwh"] = float(mean_absolute_error(y_val, pred))
            metrics["best_iteration"] = int(getattr(model, "best_iteration_", model.n_estimators))
        else:
            model.fit(X_train, y_train)

        try:
            imp = list(zip(features, model.feature_importances_.tolist()))
            imp.sort(key=lambda x: x[1], reverse=True)
            metrics["feature_importance"] = imp[:8]
        except Exception:
            pass

        d = self._panel_dir(panel_id)
        joblib.dump({"model": model, "features": features}, d / "model.joblib")
        return metrics

    def load(self, panel_id: str) -> TrainedModel:
        d = self._panel_dir(panel_id)
        obj = joblib.load(d / "model.joblib")
        return TrainedModel(model=obj["model"], features=obj["features"])

    def predict(self, trained: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
        X = feature_df[trained.features]
        pred = trained.model.predict(X)
        pred = np.clip(pred, 0, None)

        if "global_tilted_irradiance" in feature_df.columns:
            pred = np.where(feature_df["global_tilted_irradiance"].to_numpy() <= 0.5, 0.0, pred)

        return pred


def build_training_frame_idxjoin(energy: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """
    DST-safe training frame builder.
    Robustly coerces both inputs to UTC DatetimeIndex and joins on index.
    """
    e = _ensure_dtindex(energy, kind="energy")
    m = _ensure_dtindex(meteo, kind="meteo")

    # Normalize expected column names for energy
    # We expect kwh column. If HA returns "value", rename it.
    if "kwh" not in e.columns:
        if "value" in e.columns:
            e = e.rename(columns={"value": "kwh"})
        elif e.shape[1] == 1:
            e = e.rename(columns={e.columns[0]: "kwh"})

    if "kwh" not in e.columns:
        raise ValueError(f"energy must contain kwh/value column. Got columns={list(e.columns)}")

    # Join + sort
    df = e.join(m, how="inner").sort_index()

    # Build lags
    df["kwh_lag_24"] = df["kwh"].shift(24)
    if "global_tilted_irradiance" in df.columns:
        df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24)

    # Keep time column (for parts of pipeline expecting it)
    df["time"] = df.index

    df = add_time_features(df, time_col="time")
    df = df.dropna()
    return df
