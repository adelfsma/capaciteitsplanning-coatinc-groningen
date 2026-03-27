
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_VERSION = "v1.11"
HOLIDAY_FILE = os.path.join("data", "feestdagen.csv")

st.set_page_config(layout="wide", page_title="Capaciteitsplanning Coatinc Groningen")

STATUS_MAP = {
    "uitgeleverd": "Verzinkt",
    "PC Afgehaald": "Verzinkt",
    "UB V Gereed": "UB",
    "Afgehaald": "Verzinkt",
    "Gereed": "Verzinkt",
    "UB": "UB",
    "coat gereed": "Verzinkt",
    "Opgehangen": "Niet verzinkt",
    "Voorbewerking uitvoeren": "Niet verzinkt",
    "Productie gereed": "Niet verzinkt",
    "Nabewerking nog uitvoeren": "Verzinkt",
    "Ontzinkt": "Niet verzinkt",
}

NL_DAY_ABBR = {0: "ma", 1: "di", 2: "wo", 3: "do", 4: "vr", 5: "za", 6: "zo"}

def previous_workday(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d

def add_workdays(d: pd.Timestamp, days: int, holiday_dates: set) -> pd.Timestamp:
    out = pd.Timestamp(d).normalize()
    remaining = int(days)
    while remaining > 0:
        out = out + timedelta(days=1)
        if out.weekday() < 5 and out.date() not in holiday_dates:
            remaining -= 1
    return out

# Existing orders: only weekends excluded, so KG can still land on holidays.
def subtract_workdays_existing_orders(d: pd.Timestamp, days: int) -> pd.Timestamp:
    out = pd.Timestamp(d).normalize()
    remaining = int(days)
    while remaining > 0:
        out = out - timedelta(days=1)
        if out.weekday() < 5:
            remaining -= 1
    return out

def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.replace("kg", "", case=False, regex=False)
    s = s.str.replace("ton", "", case=False, regex=False)
    s = s.str.replace(r"(?<=\d)[.](?=\d{3}\b)", "", regex=True)
    s = s.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    return pd.to_numeric(s, errors="coerce")

def extract_week_label(filename: str) -> str:
    name = filename.lower().replace(".xlsx", "")
    if "+" in name:
        return "+" + name.split("+")[-1]
    if "-" in name:
        return "-" + name.split("-")[-1]
    return "onbekend"

def format_nl_axis_label(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    return f"{NL_DAY_ABBR[ts.weekday()]} {ts.day:02d}-{ts.month:02d}"

@st.cache_data(show_spinner=False)
def load_data(data_path: str):
    files = os.listdir(data_path)
    export_files = sorted([f for f in files if f.lower().startswith("export") and f.lower().endswith(".xlsx")])
    order_files = sorted([f for f in files if "orderexport2g" in f.lower() and f.lower().endswith(".xlsx")])

    if not export_files:
        raise FileNotFoundError("Geen Export+ bestanden gevonden in de gekozen map.")
    if not order_files:
        raise FileNotFoundError("Geen OrderExport2G bestand gevonden in de gekozen map.")

    export_frames = []
    export_file_summary = []

    for f in export_files:
        tmp = pd.read_excel(os.path.join(data_path, f))
        tmp["Bronbestand"] = f
        tmp["Bron_week"] = extract_week_label(f)
        export_frames.append(tmp)
        export_file_summary.append({"Bronbestand": f, "Bron_week": extract_week_label(f), "Aantal_regels_ingelezen": len(tmp)})

    export = pd.concat(export_frames, ignore_index=True)
    order = pd.read_excel(os.path.join(data_path, order_files[0]))

    export["Gewicht_export_kg"] = coerce_numeric(export["Gewicht"])
    export["Leverdatum"] = pd.to_datetime(export["Datum"], dayfirst=True, errors="coerce")
    export["Verzinkstatus"] = export["Status"].map(STATUS_MAP)
    export["Ordernummer_base"] = coerce_numeric(export["Nummer"].astype(str).str.extract(r"(\d+)")[0])

    order["Ordernummer"] = coerce_numeric(order["Ordernummer"])
    order["Gewicht_order_kg"] = coerce_numeric(order["Gewicht(ton)"]) * 1000

    row_counts = export.groupby("Ordernummer_base", dropna=False).size().rename("Regels_per_order").reset_index()

    merged = export.merge(row_counts, on="Ordernummer_base", how="left")
    merged = merged.merge(order[["Ordernummer", "Gewicht_order_kg"]], left_on="Ordernummer_base", right_on="Ordernummer", how="left")

    merged["Gewicht_2g_verdeeld_kg"] = merged["Gewicht_order_kg"] / merged["Regels_per_order"]
    merged["Gewicht_bron"] = np.where(merged["Gewicht_export_kg"].fillna(0) > 0, "Export+", "OrderExport2G verdeeld")
    merged["Gewicht_effectief_kg"] = np.where(merged["Gewicht_export_kg"].fillna(0) > 0, merged["Gewicht_export_kg"], merged["Gewicht_2g_verdeeld_kg"])

    return merged, pd.DataFrame(export_file_summary), order_files[0]

@st.cache_data(show_spinner=False)
def load_holidays():
    if not os.path.exists(HOLIDAY_FILE):
        raise FileNotFoundError("Bestand data/feestdagen.csv ontbreekt.")
    df = pd.read_csv(HOLIDAY_FILE)
    required = {"Datum","Omschrijving","Type"}
    if not required.issubset(df.columns):
        raise ValueError("feestdagen.csv moet de kolommen Datum, Omschrijving en Type bevatten.")
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date
    df = df.dropna(subset=["Datum"]).drop_duplicates(subset=["Datum","Omschrijving","Type"]).sort_values("Datum").reset_index(drop=True)
    return df

def build_horizon_and_include_holidays(start_dt: pd.Timestamp, production_workdays: int, holiday_dates: set):
    rows = []
    current = pd.Timestamp(start_dt).normalize()
    if current.weekday() >= 5:
        while current.weekday() >= 5:
            current += timedelta(days=1)

    productive_count = 0
    while productive_count < production_workdays:
        if current.weekday() < 5:
            is_holiday = current.date() in holiday_dates
            rows.append({"Verzinkdatum": current, "Is_feestdag_of_sluiting": is_holiday})
            if not is_holiday:
                productive_count += 1
        current += timedelta(days=1)
    return pd.DataFrame(rows)

def stoplight(pct: float, is_holiday: bool) -> str:
    if is_holiday:
        return "🔴"
    if pd.isna(pct):
        return ""
    if pct < 80:
        return "🟢"
    if pct <= 100:
        return "🟠"
    return "🔴"

def format_pct(x):
    if pd.isna(x):
        return ""
    return f"{x:.1f}%"

def format_int(x):
    if pd.isna(x):
        return ""
    return f"{int(round(x, 0)):,}".replace(",", ".")

def make_professional_matplotlib_chart(day_df: pd.DataFrame):
    plot_df = day_df.copy()
    labels = plot_df["Label_nl"].tolist()
    x = np.arange(len(plot_df))
    capacity = plot_df["Capaciteit_kg"].tolist()
    load = plot_df["Gewicht_kg"].tolist()
    is_holiday = plot_df["Is_feestdag_of_sluiting"].tolist()

    fig, ax = plt.subplots(figsize=(11, 4.8))

    for i, holiday in enumerate(is_holiday):
        if holiday:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.12, color="red", zorder=0)

    ax.bar(x, capacity, width=0.56, alpha=0.35, label="Capaciteit", zorder=2)
    ax.bar(x, load, width=0.36, label="Dagbelasting", zorder=3)

    ax.set_title("Capaciteit versus dagbelasting", fontsize=15, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("KG")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.legend(frameon=False, ncols=2, loc="upper left")

    ymax = max(max(capacity) if capacity else 0, max(load) if load else 0) * 1.15
    if ymax <= 0:
        ymax = 1
    ax.set_ylim(0, ymax)

    for i, val in enumerate(load):
        if val > 0:
            ax.text(i, val + ymax * 0.015, format_int(val), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig

# Sidebar logo and version
if os.path.exists("logo_coatinc_groningen.png"):
    st.sidebar.image("logo_coatinc_groningen.png", use_container_width=True)
st.sidebar.caption(APP_VERSION)

st.title("Capaciteitsplanning Coatinc Groningen")

st.sidebar.header("Instellingen")
capaciteit_ton = st.sidebar.slider("Max capaciteit per dag (ton)", 50, 90, 70, 5)
capaciteit_kg = capaciteit_ton * 1000
offset = st.sidebar.selectbox("Verzinkdatum = leverdatum - X werkdagen", [1, 2, 3, 4], index=1)
kg_per_traverse = st.sidebar.number_input("KG per traverse", min_value=100, max_value=10000, value=1000, step=100)
data_path = st.sidebar.text_input("Pad naar data", value="data")
default_start = previous_workday(date.today())
startdatum = st.sidebar.date_input("Startdatum rapport", value=default_start)
toon_alle_regels = st.sidebar.checkbox("Toon alle regels in controletab", value=True)

try:
    df_raw, export_file_summary, order_file = load_data(data_path)
    holiday_df = load_holidays()
except Exception as e:
    st.error(f"Kan data niet laden: {e}")
    st.stop()

holiday_dates = set(holiday_df["Datum"].tolist())

df = df_raw.copy()
df["Verzinkdatum"] = df["Leverdatum"].apply(lambda x: subtract_workdays_existing_orders(x, offset) if pd.notna(x) else pd.NaT)

start_ts = pd.Timestamp(startdatum).normalize()
today_ts = pd.Timestamp(date.today()).normalize()

df["Meegeteld_in_planning"] = "Nee"
df["Reden_uitsluiting"] = ""

mask_status_niet_verzinkt = df["Verzinkstatus"] == "Niet verzinkt"
mask_status_ub = df["Verzinkstatus"] == "UB"
mask_status_verzinkt = df["Verzinkstatus"] == "Verzinkt"
mask_binnen_horizon = df["Verzinkdatum"] >= start_ts

df.loc[mask_status_verzinkt, "Reden_uitsluiting"] = "Status = Verzinkt"
df.loc[mask_status_ub, "Reden_uitsluiting"] = "Status = UB"
df.loc[mask_status_niet_verzinkt & ~mask_binnen_horizon, "Reden_uitsluiting"] = "Voor startdatum rapport"
df.loc[mask_status_niet_verzinkt & mask_binnen_horizon, "Meegeteld_in_planning"] = "Ja"
df.loc[mask_status_niet_verzinkt & mask_binnen_horizon, "Reden_uitsluiting"] = ""

df_plan = df[df["Meegeteld_in_planning"] == "Ja"].copy()

horizon = build_horizon_and_include_holidays(start_ts, 10, holiday_dates)
dag_orders = df_plan.groupby("Verzinkdatum", as_index=False).agg(
    Gewicht_kg=("Gewicht_effectief_kg", "sum"),
    Aantal_orders_te_verzinken=("Nummer", "count"),
)

order_holiday_dates = df_plan[df_plan["Verzinkdatum"].dt.date.isin(holiday_dates)][["Verzinkdatum"]].drop_duplicates()
if not order_holiday_dates.empty:
    extra_holidays = order_holiday_dates.assign(Is_feestdag_of_sluiting=True)
    horizon = pd.concat([horizon, extra_holidays], ignore_index=True).drop_duplicates(subset=["Verzinkdatum"]).sort_values("Verzinkdatum").reset_index(drop=True)

dag = horizon.merge(dag_orders, on="Verzinkdatum", how="left")
dag["Gewicht_kg"] = dag["Gewicht_kg"].fillna(0.0)
dag["Aantal_orders_te_verzinken"] = dag["Aantal_orders_te_verzinken"].fillna(0).astype(int)
dag["Capaciteit_kg"] = np.where(dag["Is_feestdag_of_sluiting"], 0, capaciteit_kg)
dag["Benutting_pct"] = np.where(dag["Capaciteit_kg"] > 0, (dag["Gewicht_kg"] / dag["Capaciteit_kg"]) * 100, np.nan)
dag["Traverses_berekend"] = np.ceil(dag["Gewicht_kg"] / kg_per_traverse)
dag["Status"] = dag.apply(lambda r: stoplight(r["Benutting_pct"], r["Is_feestdag_of_sluiting"]), axis=1)
dag["Jaar"] = dag["Verzinkdatum"].dt.isocalendar().year.astype(int)
dag["Week"] = dag["Verzinkdatum"].dt.isocalendar().week.astype(int)
dag["Label_nl"] = dag["Verzinkdatum"].apply(format_nl_axis_label)
dag["Dagtype"] = np.where(dag["Is_feestdag_of_sluiting"], "Feestdag / sluiting", "Werkdag")

week = dag.groupby(["Jaar", "Week"], as_index=False).agg(
    Gewicht_kg=("Gewicht_kg", "sum"),
    Aantal_orders_te_verzinken=("Aantal_orders_te_verzinken", "sum"),
    Traverses_berekend=("Traverses_berekend", "sum"),
    Capaciteit_kg=("Capaciteit_kg", "sum"),
)
week["Benutting_pct"] = np.where(week["Capaciteit_kg"] > 0, (week["Gewicht_kg"] / week["Capaciteit_kg"]) * 100, np.nan)
week["Status"] = week["Benutting_pct"].apply(lambda x: stoplight(x, False))

def calculate_advice_date(day_df: pd.DataFrame, today_date: pd.Timestamp, holiday_dates: set):
    base_date = add_workdays(today_date, 5, holiday_dates)
    productive = day_df[(day_df["Is_feestdag_of_sluiting"] == False)].copy()
    row = productive[productive["Verzinkdatum"] == base_date]
    if not row.empty and float(row.iloc[0]["Benutting_pct"]) <= 95:
        return base_date
    later = productive[(productive["Verzinkdatum"] > base_date) & (productive["Benutting_pct"] < 95)]
    if not later.empty:
        return pd.Timestamp(later.iloc[0]["Verzinkdatum"])
    return base_date

advies_datum = calculate_advice_date(dag, today_ts, holiday_dates)

tab1, tab2, tab3 = st.tabs(["Dashboard", "Gebruikte gegevens", "Debug"])

with tab1:
    st.subheader("KPI-overzicht")
    k1, k2 = st.columns(2)
    aantal_orders_te_verzinken = int((df["Meegeteld_in_planning"] == "Ja").sum())
    totaal_kg_te_verzinken = float(df_plan["Gewicht_effectief_kg"].sum()) if len(df_plan) > 0 else 0.0
    with k1:
        st.metric("Aantal orders te verzinken", f"{aantal_orders_te_verzinken:,}".replace(",", "."))
    with k2:
        st.metric("Totaal KG te verzinken", f"{int(round(totaal_kg_te_verzinken, 0)):,}".replace(",", "."))

    st.subheader("Eerstvolgende leverdatum")
    st.markdown(
        f"""
        <div style="padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #d0d7de; background-color: #f6f8fa; margin-bottom: 0.75rem;">
            <div style="font-size: 2.2rem; font-weight: 700;">{advies_datum.strftime('%d-%m-%Y')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.pyplot(make_professional_matplotlib_chart(dag), clear_figure=True, use_container_width=True)

    st.subheader("Dagoverzicht")
    dag_display = dag.copy()
    dag_display["Verzinkdatum"] = dag_display["Verzinkdatum"].dt.date
    dag_display["Gewicht_kg"] = dag_display["Gewicht_kg"].apply(format_int)
    dag_display["Capaciteit_kg"] = dag_display["Capaciteit_kg"].apply(format_int)
    dag_display["Benutting_pct"] = dag_display["Benutting_pct"].apply(format_pct)
    dag_display["Traverses_berekend"] = dag_display["Traverses_berekend"].astype(int)
    dag_display["Dagtype"] = dag_display["Dagtype"].astype(str)

    def mark_holiday_row(row):
        if row["Dagtype"] == "Feestdag / sluiting":
            return ["background-color: rgba(220, 38, 38, 0.12)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        dag_display[
            ["Verzinkdatum", "Dagtype", "Aantal_orders_te_verzinken", "Gewicht_kg", "Capaciteit_kg", "Benutting_pct", "Traverses_berekend", "Status"]
        ].style.apply(mark_holiday_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Weekoverzicht")
    week_display = week.copy()
    week_display["Gewicht_kg"] = week_display["Gewicht_kg"].apply(format_int)
    week_display["Capaciteit_kg"] = week_display["Capaciteit_kg"].apply(format_int)
    week_display["Benutting_pct"] = week_display["Benutting_pct"].apply(format_pct)
    week_display["Traverses_berekend"] = week_display["Traverses_berekend"].astype(int)
    st.dataframe(
        week_display[["Jaar", "Week", "Aantal_orders_te_verzinken", "Gewicht_kg", "Capaciteit_kg", "Benutting_pct", "Traverses_berekend", "Status"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Feestdagen en fabriekssluiting")
    holiday_show = holiday_df.copy()
    holiday_show["Datum"] = pd.to_datetime(holiday_show["Datum"]).dt.strftime("%d-%m-%Y")
    st.dataframe(holiday_show, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Gebruikte gegevens / controletabel")
    relevant_cols = [
        "Bronbestand", "Bron_week", "Nummer", "Ordernummer_base", "Status", "Verzinkstatus",
        "Meegeteld_in_planning", "Reden_uitsluiting", "Datum", "Leverdatum", "Verzinkdatum",
        "Gewicht", "Gewicht_export_kg", "Gewicht_order_kg", "Regels_per_order",
        "Gewicht_2g_verdeeld_kg", "Gewicht_bron", "Gewicht_effectief_kg",
    ]
    relevant_cols = [c for c in relevant_cols if c in df.columns]
    controle_df = df[relevant_cols].copy() if toon_alle_regels else df_plan[relevant_cols].copy()

    if "Leverdatum" in controle_df.columns:
        controle_df["Leverdatum"] = pd.to_datetime(controle_df["Leverdatum"], errors="coerce").dt.date
    if "Verzinkdatum" in controle_df.columns:
        controle_df["Verzinkdatum"] = pd.to_datetime(controle_df["Verzinkdatum"], errors="coerce").dt.date
    for c in ["Gewicht_export_kg", "Gewicht_order_kg", "Gewicht_2g_verdeeld_kg", "Gewicht_effectief_kg"]:
        if c in controle_df.columns:
            controle_df[c] = controle_df[c].round(2)

    st.dataframe(controle_df, use_container_width=True, hide_index=True)
    csv_controle = controle_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download controletabel (CSV)", data=csv_controle, file_name="controletabel_capaciteitsplanning_v1_11.csv", mime="text/csv")

with tab3:
    st.subheader("Debug samenvatting")
    debug_rows = [
        {"Categorie": "Instellingen", "Omschrijving": "Startdatum rapport", "Waarde": str(startdatum)},
        {"Categorie": "Instellingen", "Omschrijving": "Capaciteit per dag (kg)", "Waarde": int(capaciteit_kg)},
        {"Categorie": "Instellingen", "Omschrijving": "KG per traverse", "Waarde": int(kg_per_traverse)},
        {"Categorie": "Bestanden", "Omschrijving": "Orderbestand", "Waarde": order_file},
        {"Categorie": "Kalender", "Omschrijving": "Aantal feestdagen / sluitingen", "Waarde": int(len(holiday_df))},
        {"Categorie": "Records", "Omschrijving": "Niet verzinkt", "Waarde": int((df["Verzinkstatus"] == "Niet verzinkt").sum())},
        {"Categorie": "Records", "Omschrijving": "Voor startdatum rapport", "Waarde": int(((df["Verzinkstatus"] == "Niet verzinkt") & (df["Verzinkdatum"] < start_ts)).sum())},
        {"Categorie": "Records", "Omschrijving": "Open orders", "Waarde": int((df["Meegeteld_in_planning"] == "Ja").sum())},
        {"Categorie": "Gewicht", "Omschrijving": "Totaal gewicht open orders (kg)", "Waarde": int(round(df_plan["Gewicht_effectief_kg"].sum(), 0)) if len(df_plan) > 0 else 0},
    ]
    for _, row in export_file_summary.iterrows():
        debug_rows.append({"Categorie": "Bestanden", "Omschrijving": f"{row['Bronbestand']} ({row['Bron_week']})", "Waarde": int(row["Aantal_regels_ingelezen"])})
    debug_df = pd.DataFrame(debug_rows)
    st.dataframe(debug_df, use_container_width=True, hide_index=True)
