"""
Query Translink SEQ GTFS data (data/SEQ_GTFS). Lazy-loads CSVs on first use.
Used by agent tools to answer transport queries without external API.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd

SEQ_GTFS_DIR = Path(__file__).resolve().parent.parent / "data" / "SEQ_GTFS"

_stops_df: Optional[pd.DataFrame] = None
_routes_df: Optional[pd.DataFrame] = None
_trips_df: Optional[pd.DataFrame] = None
_stop_times_df: Optional[pd.DataFrame] = None


def _ensure_stops() -> pd.DataFrame:
    global _stops_df
    if _stops_df is None:
        p = SEQ_GTFS_DIR / "stops.txt"
        if not p.exists():
            return pd.DataFrame()
        _stops_df = pd.read_csv(p)
        _stops_df["stop_lat"] = pd.to_numeric(_stops_df["stop_lat"].astype(str).str.strip(), errors="coerce")
        _stops_df["stop_lon"] = pd.to_numeric(_stops_df["stop_lon"].astype(str).str.strip(), errors="coerce")
    return _stops_df


def _ensure_routes() -> pd.DataFrame:
    global _routes_df
    if _routes_df is None:
        p = SEQ_GTFS_DIR / "routes.txt"
        if not p.exists():
            return pd.DataFrame()
        _routes_df = pd.read_csv(p)
    return _routes_df


def _ensure_trips() -> pd.DataFrame:
    global _trips_df
    if _trips_df is None:
        p = SEQ_GTFS_DIR / "trips.txt"
        if not p.exists():
            return pd.DataFrame()
        _trips_df = pd.read_csv(p)
    return _trips_df


def _ensure_stop_times() -> pd.DataFrame:
    global _stop_times_df
    if _stop_times_df is None:
        p = SEQ_GTFS_DIR / "stop_times.txt"
        if not p.exists():
            return pd.DataFrame()
        _stop_times_df = pd.read_csv(p)
    return _stop_times_df


def search_stops(query: str, limit: int = 15) -> List[dict]:
    """Search stops by name (substring, case-insensitive). Returns stop_id, stop_name, stop_lat, stop_lon."""
    df = _ensure_stops()
    if df.empty:
        return []
    q = (query or "").strip()
    if not q:
        return []
    mask = df["stop_name"].astype(str).str.lower().str.contains(q.lower(), na=False)
    out = df.loc[mask, ["stop_id", "stop_name", "stop_lat", "stop_lon"]].head(limit)
    rows = out.to_dict("records")
    for r in rows:
        r["stop_id"] = str(r.get("stop_id", ""))
    return rows


def get_departures(stop_id: str, after_time: str = "06:00", limit: int = 20) -> List[dict]:
    """
    Get next departures from a stop after the given time (HH:MM or HH:MM:SS).
    Returns list of {route_short_name, trip_headsign, departure_time}.
    """
    stop_id = str(stop_id).strip()
    if not stop_id:
        return []
    st = _ensure_stop_times()
    tr = _ensure_trips()
    rt = _ensure_routes()
    if st.empty or tr.empty or rt.empty:
        return []
    # Normalize time for comparison (stop_times use HH:MM:SS)
    t = after_time.strip()
    if len(t) == 5 and t[2] == ":":
        t = t + ":00"
    st_at_stop = st[st["stop_id"].astype(str) == stop_id].copy()
    if st_at_stop.empty:
        return []
    st_at_stop = st_at_stop[st_at_stop["departure_time"].astype(str) >= t].sort_values("departure_time").head(limit)
    merged = st_at_stop.merge(tr[["trip_id", "route_id", "trip_headsign"]], on="trip_id", how="left")
    merged = merged.merge(rt[["route_id", "route_short_name"]], on="route_id", how="left")
    merged = merged.drop_duplicates(subset=["departure_time", "route_short_name", "trip_headsign"])
    return merged[["route_short_name", "trip_headsign", "departure_time"]].head(limit).to_dict("records")
