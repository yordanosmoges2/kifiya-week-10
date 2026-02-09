import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ethiopia Financial Inclusion Forecast", layout="wide")

DATA_FILE = Path("data/processed/ethiopia_fi_unified_data_enriched.csv")
RAW_FALLBACK = Path("data/raw/ethiopia_fi_unified_data.csv")
FORECAST_FILE = Path("reports/forecast_2025_2027.csv")

# ----------------------------
# Data loaders
# ----------------------------
@st.cache_data
def load_data():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        src = str(DATA_FILE)
    elif RAW_FALLBACK.exists():
        df = pd.read_csv(RAW_FALLBACK)
        src = str(RAW_FALLBACK)
    else:
        st.error("No dataset found. Expected data/processed/ethiopia_fi_unified_data_enriched.csv or data/raw/ethiopia_fi_unified_data.csv")
        st.stop()

    # Parse date columns if present
    for c in ["observation_date", "period_start", "period_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Derived year
    if "observation_date" in df.columns:
        df["year"] = df["observation_date"].dt.year
    else:
        df["year"] = np.nan

    return df, src

@st.cache_data
def load_forecast():
    if FORECAST_FILE.exists():
        f = pd.read_csv(FORECAST_FILE)
        return f
    return None

df, data_src = load_data()
forecast = load_forecast()

# ----------------------------
# Helpers
# ----------------------------
def subset_obs(df):
    return df[df["record_type"] == "observation"].copy()

def subset_events(df):
    return df[df["record_type"] == "event"].copy()

def subset_links(df):
    return df[df["record_type"] == "impact_link"].copy()

def safe_latest_value(obs_df, indicator_code, gender="all", location="national"):
    d = obs_df[obs_df["indicator_code"] == indicator_code].copy()
    if "gender" in d.columns:
        d = d[d["gender"].fillna("all") == gender]
    if "location" in d.columns:
        d = d[d["location"].fillna("national") == location]
    d = d.dropna(subset=["observation_date", "value_numeric"]).sort_values("observation_date")
    if d.empty:
        return None
    return d.iloc[-1]

def plot_time_series(obs_df, codes, title, y_label, start_year=None, end_year=None, gender="all", location="national"):
    d = obs_df[obs_df["indicator_code"].isin(codes)].copy()
    if "gender" in d.columns:
        d = d[d["gender"].fillna("all") == gender]
    if "location" in d.columns:
        d = d[d["location"].fillna("national") == location]

    d = d.dropna(subset=["observation_date", "value_numeric"])
    if start_year is not None:
        d = d[d["year"] >= start_year]
    if end_year is not None:
        d = d[d["year"] <= end_year]

    if d.empty:
        st.info(f"No data for: {codes}")
        return

    fig = px.line(
        d.sort_values("observation_date"),
        x="observation_date",
        y="value_numeric",
        color="indicator_code",
        markers=True,
        title=title,
        labels={"observation_date": "Date", "value_numeric": y_label, "indicator_code": "Indicator"},
        hover_data=["source_name", "confidence", "notes"] if "source_name" in d.columns else None,
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_events_timeline(events_df, start_year=None, end_year=None):
    e = events_df.copy()
    if "observation_date" in e.columns:
        e = e.dropna(subset=["observation_date"])
        if start_year is not None:
            e = e[e["observation_date"].dt.year >= start_year]
        if end_year is not None:
            e = e[e["observation_date"].dt.year <= end_year]
    if e.empty:
        st.info("No events to display for selected range.")
        return

    e["event_label"] = e["indicator"].fillna(e["indicator_code"]).astype(str)
    fig = px.scatter(
        e.sort_values("observation_date"),
        x="observation_date",
        y="event_label",
        color="category" if "category" in e.columns else None,
        title="Event Timeline",
        labels={"observation_date": "Date", "event_label": "Event"},
        hover_data=["indicator_code", "source_name", "confidence", "notes"] if "source_name" in e.columns else None,
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast_fan(forecast_df, metric_prefix, title):
    # expects columns like access_base/access_p05/access_p95 etc.
    base = f"{metric_prefix}_base"
    p05 = f"{metric_prefix}_p05"
    p95 = f"{metric_prefix}_p95"
    pess = f"{metric_prefix}_pessimistic"
    opt = f"{metric_prefix}_optimistic"

    if forecast_df is None or base not in forecast_df.columns:
        st.info(f"Forecast not found for {metric_prefix}. Ensure reports/forecast_2025_2027.csv exists.")
        return

    x = forecast_df["year"]

    fig = go.Figure()
    if p05 in forecast_df.columns and p95 in forecast_df.columns and forecast_df[p05].notna().any():
        fig.add_trace(go.Scatter(x=x, y=forecast_df[p95], mode="lines", name="p95", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=x, y=forecast_df[p05], mode="lines", name="p05", fill="tonexty", line=dict(width=0), opacity=0.2))

    fig.add_trace(go.Scatter(x=x, y=forecast_df[base], mode="lines+markers", name="Base"))
    if pess in forecast_df.columns:
        fig.add_trace(go.Scatter(x=x, y=forecast_df[pess], mode="lines+markers", name="Pessimistic"))
    if opt in forecast_df.columns:
        fig.add_trace(go.Scatter(x=x, y=forecast_df[opt], mode="lines+markers", name="Optimistic"))

    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Percent (%)", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

def download_button(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Trends", "Forecasts", "Inclusion Projections", "Data & Notes"])

obs = subset_obs(df)
events = subset_events(df)
links = subset_links(df)

min_year = int(np.nanmin(obs["year"])) if obs["year"].notna().any() else 2011
max_year = int(np.nanmax(obs["year"])) if obs["year"].notna().any() else 2027

st.sidebar.markdown("### Filters")
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))
gender = st.sidebar.selectbox("Gender", ["all", "male", "female"], index=0)
location = st.sidebar.selectbox("Location", sorted(set(obs["location"].dropna().astype(str))) if "location" in obs.columns else ["national"], index=0)

# ----------------------------
# Pages
# ----------------------------
if page == "Overview":
    st.title("Ethiopia Financial Inclusion — Overview")
    st.caption(f"Data source loaded from: {data_src}")

    col1, col2, col3, col4 = st.columns(4)

    latest_acc = safe_latest_value(obs, "ACC_OWNERSHIP", gender=gender, location=location)
    latest_mm = safe_latest_value(obs, "ACC_MM_ACCOUNT", gender=gender, location=location)
    latest_cross = safe_latest_value(obs, "USG_CROSSOVER", gender=gender, location=location)
    latest_4g = safe_latest_value(obs, "ACC_4G_COV", gender=gender, location=location)

    with col1:
        st.metric("Account Ownership (ACC_OWNERSHIP)", f"{latest_acc['value_numeric']:.1f}%" if latest_acc is not None else "—",
                  help="Share of adults with an account (Findex definition).")
    with col2:
        st.metric("Mobile Money Account Rate (ACC_MM_ACCOUNT)", f"{latest_mm['value_numeric']:.2f}%" if latest_mm is not None else "—",
                  help="Findex: share of adults with a mobile money account.")
    with col3:
        st.metric("P2P/ATM Crossover Ratio (USG_CROSSOVER)", f"{latest_cross['value_numeric']:.2f}" if latest_cross is not None else "—",
                  help=">1 means P2P volume exceeds ATM withdrawals (proxy for digitization).")
    with col4:
        st.metric("4G Coverage (ACC_4G_COV)", f"{latest_4g['value_numeric']:.1f}%" if latest_4g is not None else "—",
                  help="Population coverage of 4G (enabler).")

    st.markdown("### Quick highlights")
    left, right = st.columns(2)

    with left:
        plot_time_series(
            obs,
            codes=["ACC_OWNERSHIP", "ACC_MM_ACCOUNT"],
            title="Access Indicators (Account ownership vs Mobile money account rate)",
            y_label="Percent (%)",
            start_year=year_range[0], end_year=year_range[1],
            gender=gender, location=location
        )

    with right:
        plot_time_series(
            obs,
            codes=["USG_P2P_COUNT", "USG_ATM_COUNT", "USG_CROSSOVER"],
            title="Usage Proxies (P2P vs ATM and Crossover)",
            y_label="Value",
            start_year=year_range[0], end_year=year_range[1],
            gender=gender, location=location
        )

    st.markdown("### Data download")
    download_button(df, "ethiopia_fi_unified_data_enriched.csv", "Download full dataset (CSV)")

elif page == "Trends":
    st.title("Trends Explorer")
    st.caption("Interactive time series with filters + indicator comparison view.")

    # Selector for indicators
    all_codes = sorted([c for c in obs["indicator_code"].dropna().unique()])
    default_codes = [c for c in ["ACC_OWNERSHIP", "ACC_MM_ACCOUNT", "USG_P2P_COUNT", "USG_ATM_COUNT", "ACC_4G_COV"] if c in all_codes]
    selected = st.multiselect("Choose indicators", all_codes, default=default_codes)

    plot_time_series(
        obs,
        codes=selected,
        title="Selected Indicators Over Time",
        y_label="Value",
        start_year=year_range[0], end_year=year_range[1],
        gender=gender, location=location
    )

    st.markdown("### Event timeline (overlay context)")
    plot_events_timeline(events, start_year=year_range[0], end_year=year_range[1])

    st.markdown("### Channel comparison view (P2P vs ATM counts)")
    if "USG_P2P_COUNT" in all_codes and "USG_ATM_COUNT" in all_codes:
        p2p = obs[obs["indicator_code"]=="USG_P2P_COUNT"][["year","value_numeric"]].rename(columns={"value_numeric":"p2p"})
        atm = obs[obs["indicator_code"]=="USG_ATM_COUNT"][["year","value_numeric"]].rename(columns={"value_numeric":"atm"})
        comp = pd.merge(p2p, atm, on="year", how="outer").sort_values("year")
        fig = px.line(comp, x="year", y=["p2p","atm"], markers=True, title="P2P vs ATM Transaction Counts (year-level)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("USG_P2P_COUNT or USG_ATM_COUNT not found in dataset.")

elif page == "Forecasts":
    st.title("Forecasts (2025–2027)")
    st.caption("Shows baseline + uncertainty (where available) and scenario ranges.")

    if forecast is None:
        st.warning("No forecast file found. Run Task 4 to generate: reports/forecast_2025_2027.csv")
        st.stop()

    # Model selector (simple UI; you can expand later)
    model_choice = st.selectbox("Model view", ["Base + CI", "Scenario ranges"], index=0)

    st.markdown("### ACCESS — Account Ownership forecast")
    plot_forecast_fan(forecast, "access", "ACCESS Forecast — Account Ownership Rate")

    st.markdown("### USAGE — Digital Payment Usage forecast")
    # this will still plot scenario lines even if CI is missing (p05/p95 NaN)
    plot_forecast_fan(forecast, "usage", "USAGE Forecast — Digital Payment Usage")

    st.markdown("### Forecast table")
    st.dataframe(forecast, use_container_width=True)
    download_button(forecast, "forecast_2025_2027.csv", "Download forecast table (CSV)")

elif page == "Inclusion Projections":
    st.title("Inclusion Projections & Targets")
    st.caption("Progress toward stakeholder targets with scenario selector.")

    if forecast is None:
        st.warning("No forecast file found. Run Task 4 to generate: reports/forecast_2025_2027.csv")
        st.stop()

    target = st.slider("Target for account ownership (%)", 40, 90, 60)
    scenario = st.selectbox("Scenario", ["pessimistic", "base", "optimistic"], index=1)

    # pick access scenario series
    col_map = {
        "pessimistic": "access_pessimistic",
        "base": "access_base",
        "optimistic": "access_optimistic",
    }
    access_col = col_map[scenario]

    proj = forecast[["year", access_col]].rename(columns={access_col: "projected_access"})
    proj["gap_to_target"] = target - proj["projected_access"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=proj["year"], y=proj["projected_access"], mode="lines+markers", name=f"ACCESS ({scenario})"))
    fig.add_trace(go.Scatter(x=proj["year"], y=[target]*len(proj), mode="lines", name="Target", line=dict(dash="dash")))
    fig.update_layout(title="Projected Account Ownership vs Target", xaxis_title="Year", yaxis_title="Percent (%)", yaxis_range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Progress table")
    st.dataframe(proj, use_container_width=True)

    st.markdown("### Consortium questions (auto-summary)")
    st.write(
        f"- Under the **{scenario}** scenario, projected account ownership in **2027** is "
        f"**{proj.iloc[-1]['projected_access']:.1f}%**."
    )
    if proj.iloc[-1]["projected_access"] >= target:
        st.success("Projection meets or exceeds the target by 2027.")
    else:
        st.info(f"Projection is {proj.iloc[-1]['gap_to_target']:.1f}pp below the target by 2027.")

elif page == "Data & Notes":
    st.title("Data & Notes")
    st.caption("Dataset summary, gaps, and impact links (if present).")

    st.markdown("### Dataset overview")
    st.write(df["record_type"].value_counts(dropna=False))

    st.markdown("### Confidence distribution")
    if "confidence" in df.columns:
        conf = df["confidence"].value_counts(dropna=False).reset_index()
        conf.columns = ["confidence", "count"]
        fig = px.bar(conf, x="confidence", y="count", title="Confidence Level Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sparse indicator coverage (<= 2 observations)")
    if "indicator_code" in obs.columns:
        counts = obs["indicator_code"].value_counts()
        sparse = counts[counts <= 2].reset_index()
        sparse.columns = ["indicator_code", "n_obs"]
        st.dataframe(sparse, use_container_width=True)

    st.markdown("### Impact links (Task 3)")
    if links.empty:
        st.warning("No impact_link rows found. Task 3 requires creating impact_link relationships.")
    else:
        st.dataframe(links, use_container_width=True)

    st.markdown("### Download")
    download_button(df, "ethiopia_fi_unified_data_enriched.csv", "Download full dataset (CSV)")
