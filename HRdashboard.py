import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import difflib
import re

st.title("Human Rights Dashboard")

# --- Fixed column widths ---
cols = st.columns([8, 1, 8])  # Chart 1 width doubled, Chart 2 stays the same
col1 = cols[0]
col2 = cols[2]

# --- File Uploaders ---
st.sidebar.header("Upload Portfolio & Benchmark")
portfolio_file = st.sidebar.file_uploader("Upload Portfolio (company IDs)", type=["csv", "txt"])
benchmark_file = st.sidebar.file_uploader("Upload Benchmark (company IDs)", type=["csv", "txt"])

# --- Load Data ---
chart1_data = pd.read_csv("Chart 1 data.csv")
chart2_data = pd.read_csv("Chart 2 data.csv")

# Load asset location data
asset_location_data = pd.read_csv("Asset-Location-Data.csv")

# --- Get Company IDs from Uploaded Files ---
def get_company_ids(uploaded_file):
    if uploaded_file is None:
        return []
    df = pd.read_csv(uploaded_file, header=None)
    return df.iloc[:, 0].astype(str).tolist()

portfolio_ids = get_company_ids(portfolio_file)
benchmark_ids = get_company_ids(benchmark_file)

# --- Chart 1: Impact Materiality Assessment ---
with col1:
    st.subheader("Impact Materiality Assessment")
    esrs_metrics = [col for col in chart1_data.columns if col.startswith("CSRD_ESRS")]
    esrs_metrics = sorted(esrs_metrics)

    selected_metric = st.selectbox(
        "Select ESRS Metric",
        esrs_metrics,
        index=0,
        label_visibility="visible"
    )

    if portfolio_ids and benchmark_ids:
        portfolio_df = chart1_data[chart1_data["ISSUERID"].astype(str).isin(portfolio_ids)]
        benchmark_df = chart1_data[chart1_data["ISSUERID"].astype(str).isin(benchmark_ids)]

        def percent_one(df, col):
            valid = df[col].dropna().astype(str).str.strip()
            one_count = valid.eq("1").sum()
            total = (valid != "").sum()
            return round(100 * one_count / total, 2) if total > 0 else 0

        portfolio_percent = percent_one(portfolio_df, selected_metric)
        benchmark_percent = percent_one(benchmark_df, selected_metric)

        # --- Chart area with reduced width ---
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)  # Spacer if needed
        chart_container = st.container()
        with chart_container:
            fig2, ax2 = plt.subplots(figsize=(16, 14))  # Reduced width from 48 to 16
            ax2.bar(["Portfolio", "Benchmark"], [portfolio_percent, benchmark_percent], color=["blue", "orange"])
            ax2.set_ylabel("% Companies Flagged", fontsize=40)
            ax2.set_title(f"ESRS Metric: {selected_metric}", fontsize=40, pad =140)
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='y', labelsize=40)
            ax2.tick_params(axis='x', labelsize=40)
            ax2.set_xticklabels(["Portfolio", "Benchmark"], fontsize=40)
            for i, v in enumerate([portfolio_percent, benchmark_percent]):
                ax2.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold', fontsize=40)
            fig2.tight_layout()
            st.pyplot(fig2)
    else:
        st.warning("Please upload both portfolio and benchmark files containing company IDs.")

# --- Chart 2: Fundamental Human Rights Practices ---
with col2:
    st.subheader("Fundamental Human Rights Practices")
    # Reduce the vertical gap so the bottoms of chart 1 and chart 2 align
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)  # Adjusted height for better alignment

    practice_columns = [
        "HUMAN_RGTS_POL",
        "HUMAN_RGTS_DD",
        "GRIEV_COMPLAIN_PROCEDURE_VALUE",
        "WHISTLEBLOWER_PROTECT_POLICY",
        "SUPPLIER_CODE_KEY_PROVISIONS_VALUE"
    ]

    if portfolio_ids and benchmark_ids:
        portfolio_df2 = chart2_data[chart2_data["ISSUERID"].astype(str).isin(portfolio_ids)]
        benchmark_df2 = chart2_data[chart2_data["ISSUERID"].astype(str).isin(benchmark_ids)]

        def percent_yes(df, col):
            valid = df[col].dropna().astype(str).str.strip()
            yes_count = valid.str.lower().eq("yes").sum()
            total = (valid != "").sum()
            return round(100 * yes_count / total, 2) if total > 0 else 0

        portfolio_percents = []
        benchmark_percents = []
        for col in practice_columns:
            portfolio_percents.append(percent_yes(portfolio_df2, col))
            benchmark_percents.append(percent_yes(benchmark_df2, col))

        y = np.arange(len(practice_columns))
        height = 0.35

        # Reduce the figure size to avoid PIL.Image.DecompressionBombError
        fig, ax = plt.subplots(figsize=(52, 50))  # Much smaller, but still large and readable
        ax.barh(y - height/2, portfolio_percents, height, label='Portfolio', color='blue')
        ax.barh(y + height/2, benchmark_percents, height, label='Benchmark', color='orange')

        ax.set_ylabel('Human Rights Practices', fontsize=80)
        ax.set_xlabel("Percentage of Companies with 'Yes'", fontsize=80)
        ax.set_title("Comparison of Practices: Portfolio vs Benchmark", fontsize=80, pad=160)  # Good gap below title
        ax.set_yticks(y)

        practice_labels = [
            "Human Rights Policy",
            "Human Rights Due Diligence",
            "Grievance Procedure",
            "Whistleblower Policy",
            "Supplier Code"
        ]
        ax.set_yticklabels(practice_labels, fontsize=80)
        ax.tick_params(axis='x', labelsize=80)
        ax.legend(fontsize=80)

        for i, v in enumerate(portfolio_percents):
            ax.text(v + 1, i - height/2, f"{v:.2f}%", va='center', fontweight='bold', fontsize=80)

        for i, v in enumerate(benchmark_percents):
            ax.text(v + 1, i + height/2, f"{v:.2f}%", va='center', fontweight='bold', fontsize=80)

        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Please upload both portfolio and benchmark files containing company IDs.")

# --- Chart 3: Asset Location Table ---
st.markdown("### Geospatial analysis")  # Always show the table title

# Load country score and mapping data
country_score_data = pd.read_csv("Country data.csv")
country_mapping_data = pd.read_csv("Country mapping.csv")

# In Country data.csv:
# - Column 1: Some code or ID (ignore)
# - Column 2: Country name (use for mapping)
# - Column H: GOVERNMENT_SLAVERY_OVERALL (score)

# In Country mapping.csv:
# - Column 1: ISSUER_NAME (identical to Country name in Country data.csv)
# - Column 2: Country name as found in Asset-Location-Data.csv

# Standardize country names for matching
country_score_data["Score Country"] = country_score_data.iloc[:, 1].astype(str).str.strip().str.upper()
country_score_data["Global Slavery Index (0-10 score)"] = country_score_data["GOVERNMENT_SLAVERY_OVERALL"]

country_mapping_data["Score Country"] = country_mapping_data.iloc[:, 0].astype(str).str.strip().str.upper()
country_mapping_data["Asset Country"] = country_mapping_data.iloc[:, 1].astype(str).str.strip().str.upper()

asset_location_data["Issuer Id"] = asset_location_data["Issuer Id"].astype(str).str.strip().str.upper()
asset_location_data["Country"] = asset_location_data.iloc[:, 6].astype(str).str.strip().str.upper()

# Clean country names
def clean_country(name):
    # Remove spaces, punctuation, and lowercase
    return re.sub(r'[\W_]+', '', str(name).strip().lower())

asset_location_data["Country_clean"] = asset_location_data["Country"].apply(clean_country)
country_mapping_data["Asset Country_clean"] = country_mapping_data["Asset Country"].apply(clean_country)
country_score_data["Score Country_clean"] = country_score_data["Score Country"].apply(clean_country)

if portfolio_ids:
    portfolio_ids_std = [i.strip().upper() for i in portfolio_ids]
    portfolio_assets = asset_location_data[asset_location_data["Issuer Id"].isin(portfolio_ids_std)]

    # Count assets per country for portfolio
    portfolio_country_counts = (
        portfolio_assets.groupby("Country")[asset_location_data.columns[4]]
        .count()
        .reset_index()
        .rename(columns={"Country": "Asset Country", asset_location_data.columns[4]: "Portfolio Asset Count"})
    )

    # Clean Asset Country for deduplication
    portfolio_country_counts["Asset Country Clean"] = portfolio_country_counts["Asset Country"].apply(clean_country)

    # Map Asset Country to Score Country using mapping file
    country_mapping_data["Asset Country Clean"] = country_mapping_data["Asset Country"].apply(clean_country)
    country_mapping_data["Score Country Clean"] = country_mapping_data["Score Country"].apply(clean_country)
    country_score_data["Score Country Clean"] = country_score_data["Score Country"].apply(clean_country)

    # Merge with mapping file using cleaned country names
    portfolio_country_counts = portfolio_country_counts.merge(
        country_mapping_data[["Asset Country Clean", "Score Country Clean"]],
        left_on="Asset Country Clean",
        right_on="Asset Country Clean",
        how="left"
    )

    # Fuzzy match for missing Score Country Clean
    score_country_list = country_score_data["Score Country Clean"].unique()
    def fuzzy_score_country(row):
        if pd.notnull(row["Score Country Clean"]):
            return row["Score Country Clean"]
        matches = difflib.get_close_matches(row["Asset Country Clean"], score_country_list, n=1, cutoff=0.7)
        return matches[0] if matches else row["Asset Country Clean"]

    portfolio_country_counts["Score Country Clean"] = portfolio_country_counts.apply(fuzzy_score_country, axis=1)

    # Merge with country score data using cleaned country names
    portfolio_country_counts = portfolio_country_counts.merge(
        country_score_data[["Score Country Clean", "Global Slavery Index (0-10 score)"]],
        on="Score Country Clean",
        how="left"
    )

    # Add columns for child labor and forced labor scores before merging
    country_score_data["Use of Child Labor"] = country_score_data["GOVERNMENT_USE_CHILD_LABOR"]
    country_score_data["Use of Forced Labor"] = country_score_data["GOVERNMENT_USE_FORCED_LABOR"]

    # Merge with country score data using cleaned country names, including new columns (only once)
    portfolio_country_counts = portfolio_country_counts.merge(
        country_score_data[
            ["Score Country Clean", 
             "Global Slavery Index (0-10 score)", 
             "Use of Child Labor", 
             "Use of Forced Labor"]
        ],
        left_on="Score Country Clean",
        right_on="Score Country Clean",
        how="left"
    )

    # Add original country name to portfolio_country_counts for display
    portfolio_country_counts["Asset Country Orig"] = portfolio_country_counts["Asset Country"]

    # Deduplicate by cleaned country name, sum asset counts, display original asset country name
    deduped = (
        portfolio_country_counts
        .groupby("Score Country Clean")
        .agg({
            "Asset Country Orig": "first",  # original asset country name for display
            "Portfolio Asset Count": "sum",
            "Global Slavery Index (0-10 score)_x": "first",
            "Use of Child Labor": "first",
            "Use of Forced Labor": "first"
        })
        .reset_index(drop=True)
    )

    # Filter to only countries with a score above 4
    deduped_filtered = deduped[deduped["Global Slavery Index (0-10 score)_x"] > 4]

    # Get top 20 countries by asset count (with score above 4)
    top_20 = deduped_filtered.sort_values("Portfolio Asset Count", ascending=False).head(20)
    top_20 = top_20[
        ["Asset Country Orig", "Portfolio Asset Count", "Global Slavery Index (0-10 score)_x", "Use of Child Labor", "Use of Forced Labor"]
    ].rename(
        columns={
            "Asset Country Orig": "Country",
            "Global Slavery Index (0-10 score)_x": "Global Slavery Index (0-10 score)",
            "Use of Child Labor": "Use of Child Labor",
            "Use of Forced Labor": "Use of Forced Labor"
        }
    ).reset_index(drop=True)

    st.dataframe(
        top_20,
        use_container_width=True
    )
else:
    st.warning("Please upload a portfolio file containing company IDs.")