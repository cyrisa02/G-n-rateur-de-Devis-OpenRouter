import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import tempfile
from dotenv import load_dotenv

# Charger les variables d'environnement (Streamlit Cloud lit .env à la racine)
load_dotenv()

# === CONFIGURATION OPENROUTER ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY manquant. Ajoutez-le dans le fichier `.env` ou dans les secrets de Streamlit.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# === INITIALISATION DE L'ÉTAT DE SESSION ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "prompt_metier" not in st.session_state:
    st.session_state.prompt_metier = ""
if "ready" not in st.session_state:
    st.session_state.ready = False

# === PROMPT GORK ===
GORK_PROMPT = """# RÔLE
Tu es un professionnel de la vente en B2B spécialisé dans la tapisserie d'ameublement.
Tu crées des devis détaillés basés sur les besoins des clients.
Tu es professionnel, courtois et précis.

Entreprise: SAS Tapisserie Simon
Adresse: 27 rue des Platanes, 02200 Soissons
Tarif horaire: 45€/heure

# ACCÈS AUX DONNÉES
Tu as accès aux documents suivants via le CONTEXTE:
- BD_Tissu_Casal.pdf : Prix et références tissus Casal
- BD_Tissu_Frey.pdf : Prix et références tissus Frey
- BD_Main_Oeuvre.pdf : Tarifs main d'œuvre par type de fauteuil

⚠️ IMPORTANT: Recherche TOUJOURS dans ces documents. Ne dis JAMAIS que tu n'as pas accès.

CONTEXTE DES DOCUMENTS:
{context}

HISTORIQUE DE LA CONVERSATION:
{chat_history}

QUESTION ACTUELLE:
{question}

RÉPONSE (suivre les protocoles métier et utiliser le CONTEXTE):"""

def format_chat_history(history):
    if not history:
        return "Aucun échange précédent."
    return "\n".join([f"Utilisateur: {msg['user']}\nAssistant: {msg['ai']}" for msg in history])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_user_input(user_question):
    if not st.session_state.vectorstore:
        st.warning("⚠️ Veuillez d'abord uploader les fichiers tarifs et métier.")
        return

    # Récupérer le contexte
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(user_question)
    context = "\n\n".join([d.page_content for d in docs])

    # Formater l'historique
    chat_history_str = format_chat_history(st.session_state.chat_history)

    # Construire le prompt
    full_prompt = GORK_PROMPT.format(
        context=context,
        chat_history=chat_history_str,
        question=user_question,
        prompt_metier=st.session_state.prompt_metier
    )

    # Appel au LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.2,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        max_tokens=3000
    )

    with st.spinner("Gork réfléchit..."):
        response = llm.invoke(full_prompt)

    # Sauvegarder dans l’historique
    st.session_state.chat_history.append({"user": user_question, "ai": response.content})

# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="Gork – Assistant Tapisserie", page_icon="🧵")
st.title("🧵 Gork – Assistant B2B Tapisserie d’Ameublement")
st.subheader("SAS Tapisserie Simon – Soissons")

# Sidebar
with st.sidebar:
    st.header("📁 Documents métier")
    st.write("1. **Fichiers tarifs** (BD Tissus Casal/Frey)")
    tariffs = st.file_uploader("Upload PDFs (tarifs)", type="pdf", accept_multiple_files=True, key="tariffs")
    
    st.write("2. **Prompt métier** (BD Main d’œuvre)")
    prompt_doc = st.file_uploader("Upload PDF (métier)", type="pdf", key="prompt")

    if st.button("🔄 Charger les documents"):
        if not tariffs or not prompt_doc:
            st.error("Veuillez uploader les deux types de fichiers.")
        else:
            with st.spinner("Traitement des documents..."):
                # Extraire texte tarifs
                raw_text = get_pdf_text(tariffs)
                # Splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=250,
                    length_function=len
                )
                chunks = text_splitter.split_text(raw_text)

                # Vectoriser
                st.session_state.vectorstore = get_vectorstore(chunks)

                # Extraire prompt métier
                prompt_text = get_pdf_text([prompt_doc])
                st.session_state.prompt_metier = prompt_text[:5000]  # tronquer si trop long

                st.session_state.ready = True
                st.success("✅ Documents chargés ! Vous pouvez discuter avec Gork.")

# Afficher l’historique
for msg in st.session_state.chat_history:
    with st.chat_message("human"):
        st.markdown(msg["user"])
    with st.chat_message("ai"):
        st.markdown(msg["ai"])

# Zone de saisie
if prompt := st.chat_input("Posez votre question à Gork (ex: 'Débuter une estimation')"):
    if not st.session_state.ready:
        st.warning("⚠️ Chargez d’abord les documents dans la barre latérale.")
    else:
        with st.chat_message("human"):
            st.markdown(prompt)
        handle_user_input(prompt)
        # Afficher la dernière réponse
        if st.session_state.chat_history:
            last = st.session_state.chat_history[-1]
            with st.chat_message("ai"):
                st.markdown(last["ai"])
