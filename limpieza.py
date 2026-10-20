"""
Fuel prices at the pump in Argentina (2017-2026) - data cleaning.

Source: Secretaria de Energia, "Precios en Surtidor - Resolucion 314/2016"
        http://datos.energia.gob.ar/dataset/precios-en-surtidor

Input:  datos/precios-historicos.csv   one row per price change (~3.4M rows)
        datos/precios-vigentes.csv     latest reported price per station
Output: salida/stations.csv            one row per gas station
        salida/monthly_prices.csv      price per station, product and month
        salida/current_prices.csv      latest price per station and product
        salida/data_quality_report.md  what was removed and why
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "datos"
OUT = BASE / "salida"

FIRST_VALID_DATE = pd.Timestamp("2016-06-01")  # the reporting system started in 2016
NOW = pd.Timestamp.now()
OUTLIER_LOW, OUTLIER_HIGH = 0.5, 2.0  # allowed range vs. national median of product-month
MAX_CARRY_MONTHS = 2  # months a price is still considered valid without a new report
RECENT_DAYS = 90  # a "current" price older than this is flagged as outdated
CHUNK_ROWS = 250_000

# Province as written in the source -> (display name, region).
# The source has its own "region" column, but it mixes geography with the
# Patagonian fuel-tax zone (e.g. Patagones and Malargue tagged as Patagonia)
# and has errors (Tucuman tagged as Cuyo), so the region is rebuilt from the
# province using standard geographic regions.
PROVINCES = {
    "BUENOS AIRES": ("Buenos Aires", "Pampas"),
    "CAPITAL FEDERAL": ("CABA", "Pampas"),
    "CORDOBA": ("Córdoba", "Pampas"),
    "ENTRE RIOS": ("Entre Ríos", "Pampas"),
    "LA PAMPA": ("La Pampa", "Pampas"),
    "SANTA FE": ("Santa Fe", "Pampas"),
    "CATAMARCA": ("Catamarca", "Northwest"),
    "JUJUY": ("Jujuy", "Northwest"),
    "LA RIOJA": ("La Rioja", "Northwest"),
    "SALTA": ("Salta", "Northwest"),
    "SANTIAGO DEL ESTERO": ("Santiago del Estero", "Northwest"),
    "TUCUMAN": ("Tucumán", "Northwest"),
    "CHACO": ("Chaco", "Northeast"),
    "CORRIENTES": ("Corrientes", "Northeast"),
    "FORMOSA": ("Formosa", "Northeast"),
    "MISIONES": ("Misiones", "Northeast"),
    "MENDOZA": ("Mendoza", "Cuyo"),
    "SAN JUAN": ("San Juan", "Cuyo"),
    "SAN LUIS": ("San Luis", "Cuyo"),
    "CHUBUT": ("Chubut", "Patagonia"),
    "NEUQUEN": ("Neuquén", "Patagonia"),
    "RIO NEGRO": ("Río Negro", "Patagonia"),
    "SANTA CRUZ": ("Santa Cruz", "Patagonia"),
    "TIERRA DEL FUEGO": ("Tierra del Fuego", "Patagonia"),
}

# Brand names that need more than title case.
BRANDS = {
    "YPF": "YPF",
    "SHELL C.A.P.S.A.": "Shell",
    "DAPSA S.A.": "Dapsa",
    "ESSO PETROLERA ARGENTINA S.R.L": "Esso",
    "OIL COMBUSTIBLES S.A.": "Oil",
    "BLANCA": "Unbranded",
    "SIN EMPRESA BANDERA": "Unbranded",
}


def product_name(raw):
    """Short English product name. Matching by keyword avoids accent issues."""
    s = str(raw).lower()
    if "premium" in s:
        return "Premium Gasoline"
    if "nafta" in s:
        return "Regular Gasoline"
    if "grado 3" in s:
        return "Premium Diesel"
    if "grado 2" in s:
        return "Diesel"
    if "gnc" in s:
        return "CNG"
    return None


def parse_dates(values, main_format):
    """Parse with the expected format; retry the leftovers with a flexible parser."""
    dates = pd.to_datetime(values, format=main_format, errors="coerce")
    missing = dates.isna() & values.notna()
    if missing.any():
        dates[missing] = pd.to_datetime(values[missing], format="mixed",
                                        dayfirst=True, errors="coerce")
    return dates


def drop_outliers(df, group_cols):
    """Remove prices far from the national median of the same group."""
    median = df.groupby(group_cols)["price"].transform("median")
    ok = df["price"].between(median * OUTLIER_LOW, median * OUTLIER_HIGH)
    return df[ok], int((~ok).sum())


report = {}


def log(step, rows):
    report[step] = rows
    print(f"  {step}: {rows:,}")


# ---------------------------------------------------------------------------
# 1. Read the historical file in chunks (it does not fit in memory at once)
# ---------------------------------------------------------------------------
print("1. Reading historical prices...")
cols = ["idempresa", "empresa", "direccion", "localidad", "provincia", "producto",
        "tipohorario", "precio", "fecha_vigencia", "empresabandera",
        "latitud", "longitud"]
facts, station_info = [], []
raw_rows = night_rows = 0

for chunk in pd.read_csv(RAW / "precios-historicos.csv", usecols=cols, dtype=str,
                         chunksize=CHUNK_ROWS):
    raw_rows += len(chunk)
    # Each price is reported twice (day and night shift) and they match in >99% of
    # cases, so only the day price is kept to avoid counting every price twice.
    day = chunk[chunk["tipohorario"] == "Diurno"].copy()
    night_rows += len(chunk) - len(day)

    day["date"] = parse_dates(day["fecha_vigencia"], "%d/%m/%Y %H:%M")

    facts.append(pd.DataFrame({
        "station_id": pd.to_numeric(day["idempresa"], errors="coerce"),
        "product": day["producto"].map(product_name),
        "price": pd.to_numeric(day["precio"], errors="coerce"),
        "date": day["date"],
    }))
    # Station details: keep only the most recent record of each station
    station_info.append(day.sort_values("date")
                           .drop_duplicates("idempresa", keep="last")
                           .drop(columns=["producto", "tipohorario", "precio",
                                          "fecha_vigencia"]))

facts = pd.concat(facts, ignore_index=True)
log("Raw rows in historical file", raw_rows)
log("Night-shift duplicates removed", night_rows)

# ---------------------------------------------------------------------------
# 2. Quality checks on every price change
# ---------------------------------------------------------------------------
print("2. Cleaning price changes...")
bad = facts["station_id"].isna() | facts["product"].isna()
log("Unknown station or product", int(bad.sum()))
facts = facts[~bad]

bad = facts["date"].isna()
log("Unreadable date", int(bad.sum()))
facts = facts[~bad]

bad = (facts["date"] < FIRST_VALID_DATE) | (facts["date"] > NOW)
log("Date out of range (before 2016-06 or in the future)", int(bad.sum()))
facts = facts[~bad]

bad = facts["price"].isna() | (facts["price"] <= 0)
log("Missing, zero or negative price", int(bad.sum()))
facts = facts[~bad]

before = len(facts)
facts = facts.drop_duplicates(["station_id", "product", "date"], keep="last")
log("Duplicate reports (same station, product and time)", before - len(facts))

facts["month"] = facts["date"].dt.year * 12 + facts["date"].dt.month - 1
facts, n = drop_outliers(facts, ["product", "month"])
log("Outlier prices (<50% or >200% of the national median that month)", n)
log("Valid price changes", len(facts))

# ---------------------------------------------------------------------------
# 3. Monthly table: last price of each station, product and month
# ---------------------------------------------------------------------------
# The source only records *changes*. If a station did not change its price in a
# month, its previous price was still in effect, so it is carried forward for up
# to MAX_CARRY_MONTHS months (longer gaps are treated as "no data").
print("3. Building monthly table...")
facts = facts.sort_values(["station_id", "product", "date"])
monthly = (facts.groupby(["station_id", "product", "month"], as_index=False)
                .agg(price=("price", "last"), changes=("price", "size")))

next_month = monthly.groupby(["station_id", "product"])["month"].shift(-1)
copies = ((next_month - monthly["month"]).fillna(1)
          .clip(upper=MAX_CARRY_MONTHS + 1).astype(int))
monthly = monthly.loc[monthly.index.repeat(copies)].reset_index(drop=True)
offset = monthly.groupby(["station_id", "product", "month"]).cumcount().to_numpy()
monthly["month"] = monthly["month"] + offset
monthly["changes"] = monthly["changes"].where(offset == 0, 0)
monthly["carried_forward"] = offset > 0

monthly["month"] = pd.to_datetime(pd.DataFrame({
    "year": monthly["month"] // 12, "month": monthly["month"] % 12 + 1, "day": 1}))
monthly["station_id"] = monthly["station_id"].astype(int)
log("Monthly rows (station x product x month)", len(monthly))

# ---------------------------------------------------------------------------
# 4. Current prices (latest report of each station)
# ---------------------------------------------------------------------------
print("4. Cleaning current prices...")
cur_raw = pd.read_csv(RAW / "precios-vigentes.csv", dtype=str)
cur = cur_raw[cur_raw["tipohorario"] == "Diurno"].copy()
current = pd.DataFrame({
    "station_id": pd.to_numeric(cur["idempresa"], errors="coerce"),
    "product": cur["producto"].map(product_name),
    "price": pd.to_numeric(cur["precio"], errors="coerce"),
    "last_update": parse_dates(cur["fecha_vigencia"], "%Y-%m-%d %H:%M:%S"),
})
before = len(current)
current = current.dropna()
current = current[(current["price"] > 0) & (current["last_update"] <= NOW)]
log("Current prices: invalid rows removed", before - len(current))
# Compare each price with others reported the same month: many stations stopped
# reporting long ago, so their "current" price is old, not wrong.
current["month"] = current["last_update"].dt.to_period("M")
current, n = drop_outliers(current, ["product", "month"])
current = current.drop(columns="month")
log("Current prices: outliers removed", n)
current["station_id"] = current["station_id"].astype(int)
current["days_since_update"] = (NOW - current["last_update"]).dt.days
current["is_recent"] = current["days_since_update"] <= RECENT_DAYS
log("Current prices kept", len(current))
log("Current prices updated in the last 90 days", int(current["is_recent"].sum()))

# ---------------------------------------------------------------------------
# 5. Stations table
# ---------------------------------------------------------------------------
print("5. Building stations table...")
cur_info = cur.rename(columns={"fecha_vigencia": "fv"})
cur_info["date"] = parse_dates(cur_info["fv"], "%Y-%m-%d %H:%M:%S")
info = pd.concat(station_info + [cur_info[station_info[0].columns]], ignore_index=True)
info = info.sort_values("date").drop_duplicates("idempresa", keep="last")

prov = info["provincia"].str.strip().str.upper()
stations = pd.DataFrame({
    "station_id": pd.to_numeric(info["idempresa"], errors="coerce"),
    "company": info["empresa"].str.strip(),
    "address": info["direccion"].str.strip(),
    "city": info["localidad"].str.strip().str.title(),
    "province": prov.map(lambda p: PROVINCES.get(p, (str(p).title(), None))[0]),
    "region": prov.map(lambda p: PROVINCES.get(p, (None, "Unknown"))[1]),
    "brand": info["empresabandera"].str.strip().map(
        lambda b: BRANDS.get(b, str(b).title())),
    "latitude": pd.to_numeric(info["latitud"], errors="coerce"),
    "longitude": pd.to_numeric(info["longitud"], errors="coerce"),
}).dropna(subset=["station_id"])
stations["station_id"] = stations["station_id"].astype(int)

# Coordinates outside Argentina are errors: blank them so they don't break the map
outside = ~(stations["latitude"].between(-56, -21) & stations["longitude"].between(-74, -53))
log("Stations with missing or invalid coordinates", int(outside.sum()))
stations.loc[outside, ["latitude", "longitude"]] = None
log("Stations with an unmapped province", int((stations["region"] == "Unknown").sum()))
log("Stations", len(stations))

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
print("6. Saving...")
OUT.mkdir(exist_ok=True)
stations.to_csv(OUT / "stations.csv", index=False, encoding="utf-8-sig")
monthly.to_csv(OUT / "monthly_prices.csv", index=False, encoding="utf-8-sig",
               date_format="%Y-%m-%d")
current.to_csv(OUT / "current_prices.csv", index=False, encoding="utf-8-sig",
               date_format="%Y-%m-%d %H:%M")

per_year = facts["date"].dt.year.value_counts().sort_index()
lines = ["# Data quality report", "",
         f"Generated: {NOW:%Y-%m-%d %H:%M}", "",
         "| Step | Rows |", "|---|---:|"]
lines += [f"| {k} | {v:,} |" for k, v in report.items()]
lines += ["", "## Valid price changes per year", "", "| Year | Price changes |", "|---|---:|"]
lines += [f"| {y} | {n:,} |" for y, n in per_year.items()]
(OUT / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Done. Files saved in", OUT)
