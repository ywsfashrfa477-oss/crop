from flask import Flask, request, jsonify, render_template
import pandas as pd
import math
from datetime import date, datetime

app = Flask(__name__)

# ==========================
# LOAD DATA
# ==========================

points_df = pd.read_csv("points.csv", sep="|")
soil_df = pd.read_excel("Analysis_data.xlsx")
soil_df.columns = soil_df.columns.str.strip()
points_df["id"] = points_df["id"].astype(int)
soil_df["id"] = soil_df["id"].astype(int)


# ==========================
# LOAD LSTM DATA/MODEL (ONCE)
# ==========================

final_v_df = pd.read_csv("Final_v.csv")
final_v_df["Date"] = pd.to_datetime(final_v_df["Date"], format="mixed", errors="coerce")

LSTM_SOIL_COLS = [
    "pH", "EC (ds/m)", "Ca²⁺ (ppm)", "Mg²⁺ (ppm)", "Na⁺ (ppm)",
    "K⁺ (ppm)", "CaCO₃ (%)", "ESP", "Water_TDS (ppm)",
]
LSTM_CLIMATE_COLS = ["T2M", "RH", "WS", "SWGDN"]

_soil_min = final_v_df[LSTM_SOIL_COLS].min(numeric_only=True)
_soil_max = final_v_df[LSTM_SOIL_COLS].max(numeric_only=True)
_clim_min = final_v_df[LSTM_CLIMATE_COLS].min(numeric_only=True)
_clim_max = final_v_df[LSTM_CLIMATE_COLS].max(numeric_only=True)

LSTM_OUTPUT_CROPS = ["Barley", "Cotton", "Date Palm", "Olive", "Potato", "Wheat"]

_lstm_model = None


def _minmax_scale_df(df, mn, mx):
    return (df - mn) / (mx - mn + 1e-9)


def get_lstm_model():
    global _lstm_model
    if _lstm_model is None:
        import tensorflow as tf
        _lstm_model = tf.keras.models.load_model("final version model lstm.keras")
    return _lstm_model


def build_lstm_inputs_for_auger(auger_id, when, *, depth=None):
    auger_name = f"Auger {int(auger_id)}"
    target_dt = pd.to_datetime(when)

    auger_rows = final_v_df[final_v_df["Auger"] == auger_name].copy()
    auger_rows = auger_rows.dropna(subset=["Date"]).sort_values("Date")
    if auger_rows.empty:
        return None

    if depth is None or depth not in set(auger_rows["Depth (cm)"].astype(str).unique().tolist()):
        preferred_order = ["0-10", "0-30", "0-20", "0-15", "0-25", "0 - 10", "0 - 30"]
        available = [str(x) for x in auger_rows["Depth (cm)"].unique().tolist()]
        picked = None
        for d in preferred_order:
            if d in available:
                picked = d
                break
        depth = picked if picked is not None else available[0]

    sub = auger_rows[auger_rows["Depth (cm)"] == depth].copy()
    sub = sub.dropna(subset=["Date"]).sort_values("Date")
    if sub.empty:
        return None

    target_mmdd = (int(target_dt.month), int(target_dt.day))
    mmdd = sub["Date"].apply(lambda d: (int(d.month), int(d.day)))
    exact = sub[mmdd == target_mmdd]
    if not exact.empty:
        anchor = exact.head(1)
    else:
        target_doy = int(target_dt.dayofyear)
        doy = sub["Date"].dt.dayofyear.astype(int)
        delta = (doy - target_doy).abs()
        circ = pd.concat([delta, (365 - delta).abs()], axis=1).min(axis=1)
        anchor = sub.iloc[circ.argsort()[:1]]

    anchor_date = anchor.iloc[0]["Date"]
    window = sub[sub["Date"] <= anchor_date].tail(7)
    if len(window) < 7:
        window = sub.head(7)
    if len(window) < 7:
        return None

    climate = window[LSTM_CLIMATE_COLS].astype("float32")
    soil = anchor[LSTM_SOIL_COLS].astype("float32")

    climate_s = _minmax_scale_df(climate, _clim_min, _clim_max).to_numpy(dtype="float32")[None, :, :]
    soil_s = _minmax_scale_df(soil, _soil_min, _soil_max).to_numpy(dtype="float32")

    return {
        "auger_name": auger_name,
        "depth": depth,
        "anchor_date": pd.to_datetime(anchor_date).date().isoformat(),
        "climate_input": climate_s,
        "soil_input": soil_s,
        "raw_soil": soil.iloc[0].to_dict(),
    }


def recommend_crop_lstm(auger_id, when, *, depth=None):
    built = build_lstm_inputs_for_auger(auger_id, when, depth=depth)
    if built is None:
        return None

    model = get_lstm_model()
    pred = model.predict([built["climate_input"], built["soil_input"]], verbose=0)[0]
    raw_scores = {LSTM_OUTPUT_CROPS[i]: float(pred[i]) for i in range(len(LSTM_OUTPUT_CROPS))}
    top3 = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    selected = [c for c, _ in top3]
    top3_total = sum(s for _, s in top3)
    if top3_total <= 1e-12:
        scores = {c: (1.0 / 3.0) for c in selected}
    else:
        scores = {c: (s / top3_total) for c, s in top3}

    return {
        "recommended_crops": selected,
        "score_sum": float(sum(scores.values())),
        "scores": scores,
        "anchor_date_used": built["anchor_date"],
        "depth_used": built["depth"],
        "raw_soil_used": {k: float(v) for k, v in built["raw_soil"].items()},
    }


# ==========================
# HAVERSINE DISTANCE
# ==========================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


# ==========================
# NEAREST POINTS + SOIL AGG
# ==========================

def get_nearest_points(user_lat, user_lon, *, k=3):
    rows = []
    for _, row in points_df.iterrows():
        dist = haversine(user_lat, user_lon, row["Y"], row["X"])
        rows.append({"id": int(row["id"]), "distance_km": float(dist)})
    rows.sort(key=lambda r: r["distance_km"])
    return rows[:max(1, int(k))]


def aggregate_soil_for_ids(ids):
    subset = soil_df[soil_df["id"].isin([int(x) for x in ids])].copy()
    if subset.empty:
        return None
    agg = {}
    for col in subset.columns:
        if col == "id":
            continue
        if pd.api.types.is_numeric_dtype(subset[col]):
            agg[col] = float(subset[col].mean())
        else:
            mode = subset[col].mode(dropna=True)
            agg[col] = str(mode.iloc[0]) if len(mode) else "N/A"
    return agg


# ==========================
# CROP RECOMMENDATION (RULES)
# ==========================

def _season_from_month(m):
    if m in (11, 12, 1, 2, 3):
        return "winter"
    if m in (4, 5, 6, 7, 8):
        return "summer"
    return "autumn"


def recommend_crop(soil, when):
    season = _season_from_month(int(when.month))
    ph = soil.get("pH")
    ec = soil.get("EC")
    texture = str(soil.get("Texture", "")).lower()

    crops = {
        "Wheat": 0, "Barley": 0, "Maize": 0, "Sorghum": 0,
        "Cotton": 0, "Peanut": 0, "Potato": 0, "Rice": 0,
        "Vegetables": 0, "Sugar beet": 0,
    }
    why = {k: [] for k in crops.keys()}

    if season == "winter":
        for c in ("Wheat", "Barley", "Vegetables"):
            crops[c] += 2
            why[c].append("fits winter planting window")
    elif season == "summer":
        for c in ("Maize", "Cotton", "Sorghum", "Vegetables", "Peanut"):
            crops[c] += 2
            why[c].append("fits summer planting window")
    else:
        for c in ("Vegetables", "Maize", "Potato"):
            crops[c] += 1
            why[c].append("fits autumn planting window")

    try:
        if ec is not None and float(ec) >= 4:
            for c in ("Barley", "Cotton", "Sugar beet", "Sorghum"):
                crops[c] += 3
                why[c].append("more tolerant of higher salinity (EC)")
        elif ec is not None and float(ec) >= 2:
            for c in ("Barley", "Cotton", "Vegetables"):
                crops[c] += 1
                why[c].append("moderate salinity (EC) tolerance")
        else:
            for c in ("Potato", "Peanut", "Vegetables", "Maize"):
                crops[c] += 1
                why[c].append("prefers low-to-moderate salinity (EC)")
    except Exception:
        pass

    try:
        if ph is not None and float(ph) >= 8.8:
            for c in ("Barley", "Sorghum", "Cotton"):
                crops[c] += 2
                why[c].append("more tolerant of alkaline soil (high pH)")
            crops["Rice"] -= 1
            why["Rice"].append("less suitable at very high pH")
        elif ph is not None and float(ph) >= 8.2:
            for c in ("Wheat", "Barley", "Maize", "Cotton"):
                crops[c] += 1
                why[c].append("ok in mildly alkaline soil")
        else:
            for c in ("Potato", "Vegetables", "Rice"):
                crops[c] += 1
                why[c].append("better with neutral to mildly alkaline soil")
    except Exception:
        pass

    if "sand" in texture and "loam" not in texture:
        for c in ("Peanut", "Potato", "Vegetables"):
            crops[c] += 2
            why[c].append("does well in sandy soils (drainage)")
        crops["Rice"] -= 2
        why["Rice"].append("needs heavier soil / water retention")
    elif "clay" in texture:
        for c in ("Rice", "Cotton", "Sugar beet"):
            crops[c] += 2
            why[c].append("handles heavier clay soils")
        crops["Peanut"] -= 1
        why["Peanut"].append("less ideal in heavy clay")
    elif "loam" in texture:
        for c in ("Wheat", "Maize", "Vegetables", "Cotton"):
            crops[c] += 2
            why[c].append("loam is generally versatile")

    best_crop = max(crops.items(), key=lambda kv: kv[1])[0]
    breakdown = {
        "season": season,
        "scores": crops,
        "reasons_for_best": why.get(best_crop, []),
    }
    return best_crop, breakdown


# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# NORMAL GPS ROUTE
# ==========================

@app.route("/nearest", methods=["POST"])
def nearest():
    try:
        data = request.get_json()
        user_lat = float(data["lat"])
        user_lon = float(data["lon"])

        min_distance = float("inf")
        nearest_id = None

        for _, row in points_df.iterrows():
            dist = haversine(user_lat, user_lon, row["Y"], row["X"])
            if dist < min_distance:
                min_distance = dist
                nearest_id = row["id"]

        MAX_DISTANCE_KM = 15
        if min_distance > MAX_DISTANCE_KM:
            return jsonify({
                "status": "out_of_region",
                "distance_km": round(min_distance, 2),
                "message": "You are outside the mapped region."
            })

        soil_info = soil_df[soil_df["id"] == nearest_id].fillna("N/A")
        soil_info = soil_info.to_dict(orient="records")

        return jsonify({
            "status": "found",
            "nearest_id": int(nearest_id),
            "distance_km": round(min_distance, 6),
            "soil_data": soil_info
        })

    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e), "trace": traceback.format_exc()})


# ==========================
# AUGER/ID + TODAY -> PREDICT CROP
# ==========================

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json() or {}
        k = int(data.get("k", 3))
        max_distance_km = float(data.get("max_distance_km", 15))

        when_raw = data.get("date")
        if when_raw:
            when = datetime.strptime(str(when_raw), "%Y-%m-%d").date()
        else:
            when = date.today()

        if "auger_id" in data or "id" in data:
            auger_id = int(data.get("auger_id", data.get("id")))
            point_row = points_df[points_df["id"] == auger_id]
            if point_row.empty:
                return jsonify({"status": "error", "message": f"Unknown auger_id/id: {auger_id}"}), 400
            user_lon = float(point_row.iloc[0]["X"])
            user_lat = float(point_row.iloc[0]["Y"])
            resolved_from = "auger_id"
        else:
            user_lat = float(data["lat"])
            user_lon = float(data["lon"])
            resolved_from = "gps"

        nearest_points = get_nearest_points(user_lat, user_lon, k=k)
        if not nearest_points:
            return jsonify({"status": "error", "message": "No points available."}), 500

        if nearest_points[0]["distance_km"] > max_distance_km:
            return jsonify({
                "status": "out_of_region",
                "distance_km": round(nearest_points[0]["distance_km"], 2),
                "message": "You are outside the mapped region."
            })

        used_ids = [p["id"] for p in nearest_points]
        soil_agg = aggregate_soil_for_ids(used_ids)
        if soil_agg is None:
            return jsonify({"status": "error", "message": "No soil rows found for nearest points."}), 500

        crop, details = recommend_crop(soil_agg, when)

        return jsonify({
            "status": "ok",
            "resolved_from": resolved_from,
            "input_location": {"lat": user_lat, "lon": user_lon},
            "date_used": when.isoformat(),
            "nearest_points": nearest_points,
            "soil_aggregated": soil_agg,
            "predicted_crop": crop,
            "details": details,
        })

    except KeyError as e:
        import traceback
        return jsonify({"status": "error", "message": f"Missing field: {str(e)}", "trace": traceback.format_exc()}), 400
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e), "trace": traceback.format_exc()}), 500


# ==========================
# GPS/AUGER + TODAY -> LSTM RECOMMEND CROP
# ==========================

@app.route("/predict_crop_lstm", methods=["POST"])
def predict_crop_lstm():
    try:
        data = request.get_json() or {}
        print(">>> Received data:", data)
        max_distance_km = float(data.get("max_distance_km", 15))
        print(">>> max_distance_km used:", max_distance_km)

        when_raw = data.get("date")
        when = datetime.strptime(str(when_raw), "%Y-%m-%d").date() if when_raw else date.today()
        depth = data.get("depth")

        resolved_from = None
        user_lat = None
        user_lon = None
        nearest_points = None
        auger_id = None

        if "auger_id" in data or "id" in data:
            auger_id = int(data.get("auger_id", data.get("id")))
            resolved_from = "auger_id"
        else:
            resolved_from = "gps"
            user_lat = float(data["lat"])
            user_lon = float(data["lon"])

            nearest_points = get_nearest_points(user_lat, user_lon, k=1)
            if not nearest_points:
                return jsonify({"status": "error", "message": "No points available."}), 500

            print(f">>> Nearest point distance: {nearest_points[0]['distance_km']} km, limit: {max_distance_km} km")

            if nearest_points[0]["distance_km"] > max_distance_km:
                return jsonify({
                    "status": "out_of_region",
                    "distance_km": round(nearest_points[0]["distance_km"], 2),
                    "message": "You are outside the mapped region."
                })
            auger_id = int(nearest_points[0]["id"])

        rec = recommend_crop_lstm(auger_id, when, depth=depth)
        if rec is None:
            return jsonify({
                "status": "error",
                "message": f"No matching rows in Final_v.csv for auger_id={auger_id} (expected 'Auger {auger_id}')"
            }), 400

        return jsonify({
            "status": "ok",
            "resolved_from": resolved_from,
            "input_location": None if user_lat is None else {"lat": user_lat, "lon": user_lon},
            "nearest_points": nearest_points,
            "auger_id": int(auger_id),
            "today_day": int(date.today().day),
            "today_month": int(date.today().month),
            "date_used": when.isoformat(),
            "lstm": rec,
        })

    except KeyError as e:
        import traceback
        return jsonify({"status": "error", "message": f"Missing field: {str(e)}", "trace": traceback.format_exc()}), 400
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e), "trace": traceback.format_exc()}), 500


# ==========================
# FORCE TEST FOR POINT 9
# ==========================

@app.route("/force_test_9")
def force_test_9():
    user_lat = 30.8903666826398
    user_lon = 30.9031494475213

    min_distance = float("inf")
    nearest_id = None

    for _, row in points_df.iterrows():
        dist = haversine(user_lat, user_lon, row["Y"], row["X"])
        if dist < min_distance:
            min_distance = dist
            nearest_id = row["id"]

    soil_info = soil_df[soil_df["id"] == nearest_id].fillna("N/A")
    soil_info = soil_info.to_dict(orient="records")

    return jsonify({
        "forced_test": True,
        "nearest_id": int(nearest_id),
        "distance_km": round(min_distance, 6),
        "soil_data": soil_info
    })


# ==========================
# RUN SERVER
# ==========================

if __name__ == "__main__":
    print("Server starting...")
    app.run(debug=True)
