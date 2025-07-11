
"""
Streamlit TURF Analysis Chat-like Assistant (Step-by-step UI)
-------------------------------------------------------------

This app mimics a chat experience using Streamlit's layout,
displaying one step at a time, with responses and input components below each insight.

Deployable on Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import openai

# Set page config
st.set_page_config(page_title="TURF Analysis Chat", layout="centered")

# Session state init
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.df = None
    st.session_state.summary = None

st.title("🧠 TURF Chat Assistant")

# Step 0: File Upload
if st.session_state.step == 0:
    st.markdown("**📂 Step 1: Upload your PET Excel file**")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded_file:
        st.session_state.df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded. Click Next to continue.")
        if st.button("Next"):
            st.session_state.step += 1
            st.experimental_rerun()

# Step 1: Summary of records/messages
elif st.session_state.step == 1:
    df = st.session_state.df
    message_ids = sorted(set(col.split("_")[0] for col in df.columns if col.startswith("M")))
    num_rows = df.shape[0]
    st.markdown(f"**📊 Step 2: Summary**")
    st.markdown(f"- Respondents: {num_rows}")
    st.markdown(f"- Messages: {len(message_ids)} → {', '.join(message_ids)}")
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

# Step 2: Score Stats + GPT Recommendation
elif st.session_state.step == 2:
    df = st.session_state.df
    st.markdown("**🧮 Step 3: Score Statistics**")

    summary_rows = []
    for msg in sorted(set(col.split("_")[0] for col in df.columns if col.startswith("M"))):
        for score_type in ["Differentiated", "Believable", "Motivating"]:
            col_name = f"{msg}_{score_type}"
            if col_name in df.columns:
                scores = df[col_name].dropna()
                summary_rows.append({
                    "Message": msg,
                    "ScoreType": score_type,
                    "Mean": round(scores.mean(), 2),
                    "StdDev": round(scores.std(), 2),
                    "Skew": round(scores.skew(), 2)
                })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    # Prepare GPT prompt
    prompt = "Based on these summary statistics, recommend AM or GM:\n\n"
    for _, row in summary_df.iterrows():
        prompt += f"{row['Message']} ({row['ScoreType']}): Mean={row['Mean']}, StdDev={row['StdDev']}, Skew={row['Skew']}\n"
    prompt += "\nShould we use Arithmetic Mean or Geometric Mean and why?"

    st.info("ℹ️ Sending summary to GPT for AM vs GM recommendation...")

    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a message scoring expert."},
                {"role": "user", "content": prompt}
            ]
        )
        response = gpt_response.choices[0].message.content
        st.session_state.summary = summary_df
        st.markdown(f"🧠 GPT Suggestion:\n\n{response}")

    except Exception as e:
        st.error(f"❌ GPT Error: {str(e)}")

    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

# Future: Add step 3 onward (AM/GM input, scoring, flatliner removal, binarization, TURF)
