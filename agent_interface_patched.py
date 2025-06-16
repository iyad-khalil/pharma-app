
import os
import random
import requests
from datetime import datetime, timedelta, date
from typing import Optional, Type

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.callbacks.context_callback import ContextCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
CONTEXT_API_TOKEN = st.secrets("CONTEXT_API_TOKEN", "dummy")
context_callback = ContextCallbackHandler(CONTEXT_API_TOKEN)

AZURE_CONFIG = {
    "model": "gpt-4o-mini",
    "openai_api_type": "azure",
    "openai_api_base": "https://blink-sentence-similarity.openai.azure.com/",
    "openai_api_version": "2024-02-15-preview",
    "deployment_name": "gpt-4o-mini",
    "openai_api_key": st.secrets("AZURE_OPENAI_API_KEY"),
}
MODEL_API_ENDPOINT = "http://previsionapi.blinkpharmacie.ma/predict"
PHARMACY_ID = st.secrets("PHARMACY_ID", "PHM_DEFAULT")

class SalesPredictionInput(BaseModel):
    current_ca: float = Field(description="Le chiffre d'affaires actuel de la pharmacie en MAD")

class SalesPredictionTool(BaseTool):
    name: str = "obtenir_predictions"
    description: str = "Prédit le chiffre d'affaires pour les 7 prochains jours d'une pharmacie"
    args_schema: Type[BaseModel] = SalesPredictionInput

    def _run(self, current_ca: float, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            pharmacy_id = PHARMACY_ID
            date_str = datetime.now().strftime("%Y-%m-%d")
            if current_ca <= 0:
                return " Le chiffre d'affaires doit être supérieur à 0 MAD."
            payload = {
                "pharmacy_id": pharmacy_id,
                "current_date": date_str,
                "current_ca": current_ca
            }
            headers = {"Content-Type": "application/json"}
            try:
                response = requests.post(MODEL_API_ENDPOINT, json=payload, headers=headers, timeout=30)
                if response.status_code != 200:
                    raise ValueError("API retournée non valide")
                data = response.json()
            except Exception as e:
                print(f"⚠️ API non disponible, données simulées : {e}")
                data = self._simulate_predictions(current_ca)

            st.session_state["predictions"] = data["predictions"]
            st.session_state["current_ca"] = current_ca
            return self._format_response(data, pharmacy_id, date_str, current_ca)
        except Exception as e:
            return f" Erreur: {e}"

    def _simulate_predictions(self, current_ca: float) -> dict:
        predictions = []
        base_date = datetime.now()
        for i in range(7):
            variation = random.uniform(-0.15, 0.15)
            pred_date = (base_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            predicted_ca = current_ca * (1 + variation)
            predictions.append({"date": pred_date, "predicted_ca": round(predicted_ca, 2)})
        return {"predictions": predictions}

    def _format_response(self, predictions: dict, pharmacy_id: str, date: str, current_ca: float) -> str:
        start_dt = datetime.strptime(date, "%Y-%m-%d")
        result = f"📈 PRÉDICTIONS DE VENTES - PHARMACIE {pharmacy_id}\n"
        result += f"CA de référence: {current_ca:.2f} MAD à partir du {start_dt.strftime('%d/%m/%Y')}\n"
        result += "-" * 50 + "\n"
        forecast_list = predictions.get("predictions", [])
        ca_values = []
        for forecast in forecast_list:
            date_str = datetime.strptime(forecast["date"], "%Y-%m-%d").strftime("%d/%m/%Y")
            ca_value = forecast["predicted_ca"]
            ca_values.append(ca_value)
            result += f"📅 {date_str}: {ca_value:.2f} MAD\n"
        if ca_values:
            moyenne = sum(ca_values) / len(ca_values)
            result += f"\n📊 Moyenne hebdomadaire: {moyenne:.2f} MAD"
        return result

llm = AzureChatOpenAI(
    deployment_name=AZURE_CONFIG["deployment_name"],
    openai_api_version=AZURE_CONFIG["openai_api_version"],
    openai_api_key=AZURE_CONFIG["openai_api_key"],
    azure_endpoint=AZURE_CONFIG["openai_api_base"],
    callbacks=[context_callback]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [SalesPredictionTool()]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    callbacks=[context_callback]
)

st.set_page_config(page_title="Assistant Pharma", page_icon="💊")
st.markdown("<h2>💊 Assistant Prédictif Pharma</h2>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "predictions" not in st.session_state:
    st.session_state["predictions"] = None

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Votre message..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inject today's date
    today_str = date.today().strftime("%d/%m/%Y")
    contextualized_prompt = f"Aujourd'hui, nous sommes le {today_str}. {prompt}"

    try:
        response = agent.invoke(contextualized_prompt)
        final_response = response["output"] if isinstance(response, dict) and "output" in response else str(response)
    except Exception as e:
        final_response = f"⚠️ Erreur : {e}"

    st.session_state.messages.append(AIMessage(content=final_response))
    with st.chat_message("assistant"):
        st.markdown(final_response)

if st.session_state["predictions"]:
    st.subheader("📊 Visualisation des Prédictions")
    df = pd.DataFrame(st.session_state["predictions"])
    df["date"] = pd.to_datetime(df["date"])

    st.dataframe(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["predicted_ca"], marker="o", label="Prévision")
    ax.axhline(y=st.session_state["current_ca"], color="red", linestyle="--", label="CA actuel")
    ax.set_title("Évolution du Chiffre d'Affaires")
    ax.set_ylabel("Montant (MAD)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)
