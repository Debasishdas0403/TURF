
import streamlit as st
import pandas as pd
import numpy as np
import itertools
from scipy.stats import skew
from collections import Counter

st.set_page_config(page_title="TURF Analysis Agent", layout="centered")
st.title("🧠 TURF Analysis Agent")

st.write("Upload your PET message score file (Excel), and let the agent do the work.")

# Upload file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File loaded successfully!")

    # Message and respondent info
    num_rows = df.shape[0]
    message_ids = sorted(set(col.split("_")[0] for col in df.columns if col.startswith("M")))
    st.write(f"**📄 Respondents:** {num_rows}")
    st.write(f"**💬 Messages:** {len(message_ids)} → {', '.join(message_ids)}")

    # Effectiveness scoring method
    method = st.radio("Choose Effectiveness Method", ["AM", "GM"])

    # Score summary
    score_types = ['Differentiated', 'Believable', 'Motivating']
    summary_rows = []
    for msg in message_ids:
        for score_type in score_types:
            col_name = f"{msg}_{score_type}"
            if col_name in df.columns:
                scores = df[col_name].dropna()
                summary_rows.append({
                    "Message": msg,
                    "ScoreType": score_type,
                    "Mean": round(scores.mean(), 2),
                    "StdDev": round(scores.std(), 2),
                    "Skew": round(skew(scores), 2)
                })
    summary_df = pd.DataFrame(summary_rows)

    st.subheader("📊 Score Summary (Mean, StdDev, Skew)")
    st.dataframe(summary_df)

    # Calculate effectiveness
    effectiveness_df = pd.DataFrame()
    for msg in message_ids:
        cols = [f"{msg}_{s}" for s in score_types]
        if method == "AM":
            effectiveness_df[f"{msg}_Effectiveness"] = df[cols].mean(axis=1)
        else:
            effectiveness_df[f"{msg}_Effectiveness"] = df[cols].replace(0.01, np.nan).prod(axis=1)**(1/3)

    # Flatliner removal
    st.subheader("🧹 Flatliner Removal")
    remove_flatliners = st.radio("Remove Flatliners?", ["Yes", "No"])
    variance_threshold = st.number_input("Variance Threshold", value=0.05)

    if remove_flatliners == "Yes":
        row_vars = effectiveness_df.var(axis=1)
        effectiveness_df = effectiveness_df[row_vars >= variance_threshold]
        st.write(f"Records after flatliner removal: {len(effectiveness_df)}")

    # Binarization
    st.subheader("🧮 Binarization Method")
    bin_method = st.radio("Choose Method", ["T2B", "Index (5% above mean)"])

    if bin_method == "T2B":
        binarized_df = effectiveness_df.applymap(lambda x: 1 if x > 5 else 0)
    else:
        binarized_df = effectiveness_df.copy()
        for i, row in effectiveness_df.iterrows():
            t = row.mean() * 1.05
            binarized_df.loc[i] = row.apply(lambda x: 1 if x >= t else 0)

    # TURF Analysis
    st.subheader("📈 TURF Analysis")
    results = []
    best_combos = {}
    for k in range(1, 6):
        combos = list(itertools.combinations(binarized_df.columns, k))
        best = max(combos, key=lambda c: (binarized_df[list(c)].sum(axis=1) > 0).mean())
        reach = (binarized_df[list(best)].sum(axis=1) > 0).mean()
        results.append((k, round(reach*100, 2), [m.split('_')[0] for m in best]))
        best_combos[k] = best

    for k, reach, combo in results:
        st.write(f"**{k} messages** → Reach: {reach}% → {', '.join(combo)}")

    # Monte Carlo
    st.subheader("🎲 Monte Carlo Simulation")
    run_mc = st.checkbox("Run Monte Carlo Simulation?")
    if run_mc:
        bundle_size = st.slider("Message Bundle Size", 1, 5, 3)
        iterations = st.slider("Number of Iterations", 10, 100, 25)
        winning_combos = []

        for i in range(iterations):
            sample = binarized_df.sample(frac=0.8, random_state=i)
            combos = list(itertools.combinations(binarized_df.columns, bundle_size))
            best = max(combos, key=lambda c: (sample[list(c)].sum(axis=1) > 0).mean())
            winning_combos.append(tuple(sorted(best)))

        combo_counts = Counter(winning_combos)
        most_common = combo_counts.most_common(5)

        st.write("Top Winning Combinations:")
        for combo, freq in most_common:
            st.write(f"{', '.join([m.split('_')[0] for m in combo])} → {freq} wins")

        try:
            if tuple(sorted(best_combos[bundle_size])) in combo_counts:
                st.success("✅ Stable: Most frequent combo matches TURF recommendation.")
            else:
                st.warning("⚠️ Unstable: Monte Carlo result differs from TURF best combo.")
        except:
            st.warning("⚠️ Could not validate against TURF. Rerun Phase 4 first.")
