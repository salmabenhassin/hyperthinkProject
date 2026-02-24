import streamlit as st
import requests

st.set_page_config(page_title="Attention Is All You Need Bot", layout="centered")
st.title("📘 Chat with the Paper")
st.caption("Document source : Attention Is All You Need (Vaswani et al., 2017)")

# Gestion session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input utilisateur
if prompt := st.chat_input("Posez une question technique sur le Transformer..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultation de l'index FAISS..."):
            try:
                # Appel API
                res = requests.post("http://localhost:8000/query", json={"q": prompt})
                
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(data["answer"])
                    
                    # Afficher les sources proprement
                    with st.expander("Voir les passages sources"):
                        for s in data["sources"]:
                            st.markdown(f"**Score {s['score']:.2f}** : {s['preview']}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
                else:
                    st.error("Erreur API. Vérifiez que main.py tourne.")
            except Exception as e:
                st.error(f"Connexion échouée : {e}")