import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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

# --- Get Company IDs from Uploaded Files ---
def get_company_ids(uploaded_file):
    if uploaded_file is None:
        return []
    df = pd.read_csv(uploaded_file, header=None)
    # Standardize: strip spaces and uppercase
    return df.iloc[:, 0].astype(str).str.strip().str.upper().tolist()

# Standardize ISSUERID in both dataframes
chart1_data["ISSUERID"] = chart1_data["ISSUERID"].astype(str).str.strip().str.upper()
chart2_data["ISSUERID"] = chart2_data["ISSUERID"].astype(str).str.strip().str.upper()

portfolio_ids = get_company_ids(portfolio_file)
benchmark_ids = get_company_ids(benchmark_file)

# --- Debugging Info ---
# Remove display of Portfolio IDs and Benchmark IDs
# st.write("Portfolio IDs:", portfolio_ids)
# st.write("Benchmark IDs:", benchmark_ids)
# st.write("Sample ISSUERIDs:", chart1_data["ISSUERID"].unique()[:10])

# Keep only the row counts
st.write("Portfolio rows:", len(chart1_data[chart1_data["ISSUERID"].isin(portfolio_ids)]))
st.write("Benchmark rows:", len(chart1_data[chart1_data["ISSUERID"].isin(benchmark_ids)]))

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
        portfolio_df = chart1_data[chart1_data["ISSUERID"].isin(portfolio_ids)]
        benchmark_df = chart1_data[chart1_data["ISSUERID"].isin(benchmark_ids)]

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
        portfolio_df2 = chart2_data[chart2_data["ISSUERID"].isin(portfolio_ids)]
        benchmark_df2 = chart2_data[chart2_data["ISSUERID"].isin(benchmark_ids)]

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
        ax.set_title("Comparison of Practices: Portfolio vs Benchmark", fontsize=80, pad=190)  # Good gap below title
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