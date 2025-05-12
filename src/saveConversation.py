import pandas as pd
from datetime import datetime
import streamlit as st
import os

def save_conversation(title, history)->None:
    user_conversation = ""
    bot_conversation = ""
    for chat in history:
        user_conversation += chat['user']
        bot_conversation += chat['bot']
    
    data = {
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Titre": [title],
        "user": [user_conversation],
        "bot": [bot_conversation]
    }
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname("data/conversations.csv"), exist_ok=True)
    try:
        # Append mode ('a'), header only if the file does not already exist
        df.to_csv("data/conversations.csv", mode='a', header=not pd.io.common.file_exists("data/conversations.csv"), sep=";", index=False)
        st.success("Conversation sauvegardée avec succès !")
    except FileNotFoundError as e:
        st.error(f"Erreur : fichier introuvable - {e}")
    except PermissionError as e:
        st.error(f"Erreur : permission refusée - {e}")
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde de la conversation: {e}")
        
def load_conversations()->pd.DataFrame:
    try:
        return pd.read_csv("data/conversations.csv", sep=";")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Titre", "Conversation"])

def delete_conversation(title)->None:
    try:
        df = load_conversations()
        df = df[df["Titre"] != title]
        df.to_csv("data/conversations.csv", index=False, sep=";")
        st.sidebar.success("Conversation supprimée avec succès !")
    except Exception as e:
        st.sidebar.error(f"Erreur lors de la suppression de la conversation: {e}")