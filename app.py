import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import markdown
import logging
from datetime import datetime

# Importations CORRIGÉES pour LangChain
from langchain_huggingface import HuggingFaceEmbeddings  # CORRIGÉ
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI  # CORRIGÉ
# ✅ Nouvelle approche (à adapter selon votre logique)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration des dossiers
UPLOAD_FOLDER = 'data'
TARIFFS_FOLDER = os.path.join(UPLOAD_FOLDER, 'tarifs')
METIER_FOLDER = os.path.join(UPLOAD_FOLDER, 'metier')

os.makedirs(TARIFFS_FOLDER, exist_ok=True)
os.makedirs(METIER_FOLDER, exist_ok=True)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    TARIFFS_FOLDER=TARIFFS_FOLDER,
    METIER_FOLDER=METIER_FOLDER,
    MAX_CONTENT_LENGTH=50 * 1024 * 1024  # 50MB
)

# Configuration OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY manquant dans .env")

# CONFIGURATION CORRECTE pour OpenRouter
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Variables globales
knowledge_base = None
prompt_metier_doc = None
conversation_chain = None
chat_history = []
documents_loaded = False

# PROMPT GORK INTÉGRÉ
GORK_SYSTEM_PROMPT = """# RÔLE
Tu es un professionnel de la vente en B2B spécialisé dans la tapisserie d'ameublement.
Tu crées des devis détaillés basés sur les besoins des clients.
Ton professionnel, courtois et précis.

Entreprise: SAS Tapisserie Simon
Adresse: 27 rue des Platanes, 02200 Soissons
Tarif horaire: 45€/heure

# ACCÈS AUX DONNÉES
Tu as accès aux documents suivants via le CONTEXTE:
- BD_Tissu_Casal.pdf : Prix et références tissus Casal
- BD_Tissu_Frey.pdf : Prix et références tissus Frey
- BD_Main_Oeuvre.pdf : Tarifs main d'œuvre par type de fauteuil

⚠️ IMPORTANT: Recherche TOUJOURS dans ces documents. Ne dis JAMAIS que tu n'as pas accès.

# PROCESSUS (4 PROTOCOLES)

## PROTOCOLE 1: RÉSUMÉ DE PROJET
Format de sortie:
```
Nom du client: [À remplir]
ID du projet: [À remplir]
Adresse du client: [À remplir]
Description du travail: [À remplir]
```

Puis demande: "Veux-tu que je passe par le 'Protocole de matériel' ou as-tu déjà les matériaux ?"

## PROTOCOLE 2: PROTOCOLE DE MATÉRIEL
Pose des questions pour déterminer:
- Type de fauteuil (Voltaire, Bergère, etc.)
- Type de travail: Réfection complète OU changement tissu
- Tissu: uni ou avec motif? (IMPORTANT pour calcul)
- Client fournit le tissu? (majoration 20% si OUI)
- Finition: Sans / Clous décoratifs / Galon / Passepoil
- Nombre de pièces

RÈGLES:
- Réfection complète = Garniture Crins et matières naturelles
- Tissu motif = Consommation différente (voir calculs)

## PROTOCOLE 3: CALCUL
### A. MAIN D'ŒUVRE (BD_Main_Oeuvre.pdf)
- Réfection complète: `Prix_unitaire_réfection_totale`
- Changement tissu: `Prix_unitaire_recouvrement`
- Client fournit tissu: +20% majoration

### B. CONSOMMATION TISSU
**Tissu UNI:**
- 1 fauteuil: `Consommation_tissu_1_fauteuil`
- 2 fauteuils: `Consommation_tissu_2_fauteuils`

**Tissu MOTIF:**
- 1 fauteuil: `Consommation_tissu_1_fauteuil × 1,3 + Motif/Raccord`
- 2 fauteuils: `Consommation_tissu_2_fauteuils × 1,3 + Motif/Raccord`

### C. PRIX TISSUS (BD_Tissu)
- Prix unitaire × Consommation × 1,50 (marge 50%)

### D. DÉPLACEMENTS
- Kilomètres × 0,8€

## PROTOCOLE 4: DEVIS FINAL

```markdown
# DEVIS - [Type de Projet]

**SAS Tapisserie Simon**
27 rue des Platanes
02200 Soissons
Date: {date}

## 📋 INFORMATIONS CLIENT
- **Nom:** [Nom]
- **Adresse:** [Adresse]
- **Référence:** [ID]

## 📝 DESCRIPTION
[Description détaillée]

## 💰 PRESTATIONS

### Main d'œuvre - [Type fauteuil]
- **Type:** [Réfection / Changement]
- **Quantité:** [X]
- **Tarif unitaire:** [Prix] € HT
- **Total:** [Calcul] € HT

### Fournitures
#### Tissu
- **Référence:** [Réf BD]
- **Nom:** [Nom tissu]
- **Type:** [Uni / Motif raccord X cm]
- **Consommation:** [X,X] m
- **Prix unitaire:** [Prix] € HT/m
- **Total:** [Calcul] € HT

#### Finition
- **Type:** [Sans/Clous/Galon/Passepoil]
- **Coût:** [X] € HT

### Déplacements
- **Kilométrage:** [X] km
- **Indemnité:** [X] € HT

## 📊 RÉCAPITULATIF

| Désignation | Montant HT |
|-------------|------------|
| Main d'œuvre | [X] € |
| Fournitures | [X] € |
| Déplacements | [X] € |
| **Sous-total HT** | **[X] €** |
| TVA (20%) | [X] € |
| **TOTAL TTC** | **[X] €** |

## 📅 CONDITIONS
- **Délai:** [X] jours
- **Paiement:** [Modalités]
- **Validité:** 30 jours

*Devis établi le {date}*
```

# COMMANDES
- "Débuter une estimation" : Lance Protocole 1
- "Donne moi les coûts des matériaux en direct" : Recherche prix dans BD

{prompt_metier}

CONTEXTE DES DOCUMENTS:
{context}

QUESTION: {question}

RÉPONSE (suivre protocoles et utiliser CONTEXTE):"""


def initialize_embeddings():
    """Initialise les embeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ Embeddings initialisés")
        return embeddings
    except Exception as e:
        logger.error(f"Erreur embeddings: {e}")
        raise


def process_pdfs(file_paths):
    """Charge et vectorise les PDFs."""
    documents = []
    
    for file_path in file_paths:
        try:
            logger.info(f"📄 Chargement: {os.path.basename(file_path)}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Ajouter métadonnées
            for doc in docs:
                doc.metadata['source_file'] = os.path.basename(file_path)
                if 'Casal' in file_path:
                    doc.metadata['type'] = 'BD_TISSU_CASAL'
                elif 'Frey' in file_path:
                    doc.metadata['type'] = 'BD_TISSU_FREY'
                elif 'Main' in file_path or 'Oeuvre' in file_path:
                    doc.metadata['type'] = 'BD_MAIN_OEUVRE'
                else:
                    doc.metadata['type'] = 'TARIF'
            
            documents.extend(docs)
            logger.info(f"  ✓ {len(docs)} pages chargées")
            
        except Exception as e:
            logger.error(f"  ✗ Erreur {os.path.basename(file_path)}: {e}")

    if not documents:
        logger.warning("Aucun document chargé")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"✂️  {len(chunks)} chunks créés")
        
        embeddings = initialize_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info(f"✓ Vectorstore créé ({len(chunks)} chunks)")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Erreur création vectorstore: {e}")
        return None


def get_conversation_chain(vectorstore, prompt_metier_content=""):
    """Initialise la chaîne de conversation avec Gork."""
    global conversation_chain, chat_history
    
    try:
        # LLM OpenRouter CORRECTEMENT configuré
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.2,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            max_tokens=3000
        )
        logger.info("✓ LLM initialisé")
        
        # Mémoire
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Prompt avec Gork
        CUSTOM_PROMPT = PromptTemplate(
            template=GORK_SYSTEM_PROMPT,
            input_variables=["context", "question", "prompt_metier"]
        )
        
        partial_prompt = CUSTOM_PROMPT.partial(prompt_metier=prompt_metier_content)
        
        # Chaîne conversationnelle
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": partial_prompt},
            return_source_documents=True,
            verbose=True
        )
        
        logger.info("✓ Chaîne conversationnelle Gork initialisée")
        return conversation_chain
        
    except Exception as e:
        logger.error(f"Erreur création chaîne: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def index():
    """Page d'accueil."""
    try:
        tarif_files = [f for f in os.listdir(app.config['TARIFFS_FOLDER']) if f.endswith('.pdf')]
        metier_files = [f for f in os.listdir(app.config['METIER_FOLDER']) if f.endswith('.pdf')]
        prompt_file = metier_files[0] if metier_files else None
        
        return render_template('index.html', 
                             tarif_files=tarif_files, 
                             prompt_file=prompt_file,
                             documents_loaded=documents_loaded)
    except Exception as e:
        logger.error(f"Erreur page d'accueil: {e}")
        return render_template('index.html', tarif_files=[], prompt_file=None, documents_loaded=False)


@app.route('/upload_tarif', methods=['POST'])
def upload_tarif():
    """Upload fichiers tarifs."""
    global knowledge_base, conversation_chain, documents_loaded
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Fichier vide'}), 400
        
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['TARIFFS_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"📄 Tarif uploadé: {filename}")
            
            # Recharger tous les tarifs
            tarif_files = [
                os.path.join(app.config['TARIFFS_FOLDER'], f) 
                for f in os.listdir(app.config['TARIFFS_FOLDER']) 
                if f.endswith('.pdf')
            ]
            
            knowledge_base = process_pdfs(tarif_files)
            
            if knowledge_base and prompt_metier_doc:
                conversation_chain = get_conversation_chain(knowledge_base, prompt_metier_doc)
                documents_loaded = True
            
            return jsonify({
                'message': f'✓ Fichier uploadé: {filename}',
                'filename': filename,
                'documents_loaded': documents_loaded
            }), 200
        
        return jsonify({'error': 'Type de fichier invalide'}), 400
        
    except Exception as e:
        logger.error(f"Erreur upload tarif: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload_prompt_metier', methods=['POST'])
def upload_prompt_metier():
    """Upload fichier métier."""
    global prompt_metier_doc, conversation_chain, documents_loaded
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Fichier vide'}), 400
        
        if file and file.filename.endswith('.pdf'):
            # Supprimer ancien
            for f in os.listdir(app.config['METIER_FOLDER']):
                os.remove(os.path.join(app.config['METIER_FOLDER'], f))
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['METIER_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"📄 Métier uploadé: {filename}")
            
            # Charger contenu
            loader = PyPDFLoader(filepath)
            prompt_metier_doc = " ".join([doc.page_content for doc in loader.load()])
            
            if knowledge_base:
                conversation_chain = get_conversation_chain(knowledge_base, prompt_metier_doc)
                documents_loaded = True
            
            return jsonify({
                'message': f'✓ Prompt métier uploadé: {filename}',
                'filename': filename,
                'documents_loaded': documents_loaded
            }), 200
        
        return jsonify({'error': 'Type de fichier invalide'}), 400
        
    except Exception as e:
        logger.error(f"Erreur upload métier: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat avec Gork."""
    global chat_history, conversation_chain, knowledge_base, prompt_metier_doc, documents_loaded

    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message vide'}), 400

        if not knowledge_base or not prompt_metier_doc:
            return jsonify({
                'response': "⚠️ Veuillez d'abord uploader:\n1. Fichiers tarifs (BD Tissus)\n2. Prompt métier (BD Main d'œuvre)"
            }), 503

        if not conversation_chain:
            conversation_chain = get_conversation_chain(knowledge_base, prompt_metier_doc)
            if not conversation_chain:
                return jsonify({'response': "❌ Erreur initialisation IA"}), 500

        logger.info("=" * 80)
        logger.info(f"💬 Question: {user_message}")
        logger.info(f"📝 Historique: {len(chat_history)} échanges")

        # APPEL CORRECT avec invoke()
        response = conversation_chain.invoke({'question': user_message})
        ai_response = response['answer']
        source_docs = response.get('source_documents', [])
        
        # Log sources
        if source_docs:
            logger.info(f"📚 Sources: {len(source_docs)} docs")
            for i, doc in enumerate(source_docs[:3], 1):
                source = doc.metadata.get('source_file', 'Unknown')
                logger.info(f"  {i}. {source}")
        
        chat_history.append({"user": user_message, "ai": ai_response})
        
        # Convertir en HTML
        html_response = markdown.markdown(ai_response)
        
        logger.info(f"✅ Réponse: {len(ai_response)} car.")
        logger.info("=" * 80)
        
        return jsonify({
            'response': ai_response,
            'html_response': html_response,
            'sources_used': len(source_docs),
            'chat_history_length': len(chat_history)
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f"Erreur: {str(e)}",
            'response': f"⚠️ Une erreur est survenue.\n\nDétails: {str(e)}"
        }), 500


@app.route('/download_markdown', methods=['POST'])
def download_markdown():
    """Télécharge le devis."""
    try:
        markdown_content = request.json.get('content')
        if not markdown_content:
            return jsonify({'error': 'Aucun contenu'}), 400

        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"devis_tapisserie_{date_str}.md"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"💾 Devis téléchargé: {filename}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Erreur téléchargement: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/delete_tarif/<filename>', methods=['DELETE'])
def delete_tarif(filename):
    """Supprime un fichier tarif."""
    global knowledge_base, conversation_chain, documents_loaded
    
    try:
        filepath = os.path.join(app.config['TARIFFS_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️  Supprimé: {filename}")
            
            tarif_files = [
                os.path.join(app.config['TARIFFS_FOLDER'], f)
                for f in os.listdir(app.config['TARIFFS_FOLDER'])
                if f.endswith('.pdf')
            ]
            
            if not tarif_files:
                knowledge_base = None
                conversation_chain = None
                documents_loaded = False
            else:
                knowledge_base = process_pdfs(tarif_files)
                if prompt_metier_doc:
                    conversation_chain = get_conversation_chain(knowledge_base, prompt_metier_doc)
            
            return jsonify({'message': f'Fichier {filename} supprimé'}), 200
        
        return jsonify({'error': 'Fichier non trouvé'}), 404
        
    except Exception as e:
        logger.error(f"Erreur suppression: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/check_status', methods=['GET'])
def check_status():
    """Statut du système."""
    tarifs_count = len([f for f in os.listdir(app.config['TARIFFS_FOLDER']) if f.endswith('.pdf')])
    metier_count = len([f for f in os.listdir(app.config['METIER_FOLDER']) if f.endswith('.pdf')])
    
    return jsonify({
        'system': 'Gork Tapisserie',
        'documents_loaded': documents_loaded,
        'knowledge_base_ready': knowledge_base is not None,
        'conversation_chain_ready': conversation_chain is not None,
        'tarifs_files': tarifs_count,
        'metier_files': metier_count,
        'chat_history_length': len(chat_history)
    })


@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Réinitialise la conversation."""
    global chat_history
    chat_history = []
    logger.info("🔄 Conversation réinitialisée")
    return jsonify({'message': 'Conversation réinitialisée'})


@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'system': 'Gork Tapisserie Assistant',
        'version': '2.0.0'
    })


# Initialisation au démarrage
with app.app_context():
    try:
        logger.info("=" * 80)
        logger.info("🚀 DÉMARRAGE SYSTÈME GORK")
        logger.info("=" * 80)
        
        # Charger fichiers existants
        tarif_files_existing = [
            os.path.join(app.config['TARIFFS_FOLDER'], f)
            for f in os.listdir(app.config['TARIFFS_FOLDER'])
            if f.endswith('.pdf')
        ]
        
        if tarif_files_existing:
            knowledge_base = process_pdfs(tarif_files_existing)
        
        prompt_file_existing = [
            f for f in os.listdir(app.config['METIER_FOLDER'])
            if f.endswith('.pdf')
        ]
        
        if prompt_file_existing:
            filepath = os.path.join(app.config['METIER_FOLDER'], prompt_file_existing[0])
            loader = PyPDFLoader(filepath)
            prompt_metier_doc = " ".join([doc.page_content for doc in loader.load()])

        if knowledge_base and prompt_metier_doc:
            conversation_chain = get_conversation_chain(knowledge_base, prompt_metier_doc)
            documents_loaded = True
            logger.info("✅ Système prêt")
        else:
            logger.info("⚠️  En attente des fichiers")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Erreur initialisation: {e}")


if __name__ == '__main__':
    logger.info("🌐 Serveur Flask sur http://127.0.0.1:5000")

    app.run(debug=True, host='127.0.0.1', port=5000)

