
"""
TURF Analysis Chat Assistant (Gradio UI)
----------------------------------------

This script creates a web-based step-by-step TURF Analysis assistant using Gradio's ChatInterface.
It simulates a guided GPT-like flow, allowing users to:

1. Upload a PET Excel file
2. Summarize the structure
3. Get GPT recommendation for AM vs GM
4. Continue analysis (future steps to be added)

You can run this with: python app.py
"""

import gradio as gr
import pandas as pd
import numpy as np
from io import BytesIO
import openai

# Insert your OpenAI API Key here
openai.api_key = "sk-REPLACE_THIS_WITH_YOUR_KEY"

# Global session state to track steps and data
session = {"step": 0, "df": None, "summary": None}

def turf_chat(user_input, file=None):
    global session

    # Step 0: Ask for file upload
    if session["step"] == 0:
        if file:
            session["df"] = pd.read_excel(file.name)
            session["step"] += 1
            return "✅ File uploaded successfully. Click 'Next' to summarize the data."
        else:
            return "📂 Please upload a PET Excel file to begin."

    # Step 1: Summary of respondents and messages
    if session["step"] == 1:
        df = session["df"]
        message_ids = sorted(set(col.split("_")[0] for col in df.columns if col.startswith("M")))
        num_rows = df.shape[0]
        response = f"📊 Data Summary:\n- Respondents: {num_rows}\n- Messages: {len(message_ids)} → {', '.join(message_ids)}"
        session["step"] += 1
        return response + "\n\n▶️ Click Next to analyze message score distribution."

    # Step 2: Summary stats + GPT recommendation (AM vs GM)
    if session["step"] == 2:
        df = session["df"]
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
        session["summary"] = summary_df

        # Build GPT prompt
        prompt = "Based on these summary statistics, recommend AM or GM:\n\n"
        for _, row in summary_df.iterrows():
            prompt += f"{row['Message']} ({row['ScoreType']}): Mean={row['Mean']}, StdDev={row['StdDev']}, Skew={row['Skew']}\n"
        prompt += "\nShould we use Arithmetic Mean or Geometric Mean and why?"

        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a message scoring expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            session["step"] += 1
            return "🧠 GPT Recommendation:\n" + gpt_response.choices[0].message.content + "\n\nPlease select AM or GM and continue."
        except Exception as e:
            return f"❌ GPT Error: {str(e)}"

    return "🎉 End of demo flow. Add more steps in app.py to continue."

# Launch Gradio Chat Interface
chatbot = gr.ChatInterface(fn=turf_chat,
                           title="🧠 TURF Chat Assistant",
                           chatbot=gr.Chatbot(),
                           textbox=gr.Textbox(placeholder="Type 'Next' to continue..."),
                           additional_inputs=[gr.File(label="Upload PET Excel")],
                           theme="soft")

if __name__ == "__main__":
    chatbot.launch()
