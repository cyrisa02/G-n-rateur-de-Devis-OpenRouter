import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import io

app = Flask(__name__)

# Configuration des dossiers d'upload
UPLOAD_FOLDER = 'uploads'
TARIFFS_FOLDER = os.path.join(UPLOAD_FOLDER, 'tariffs')
PROMPT_FOLDER = os.path.join(UPLOAD_FOLDER, 'prompt_metier')

os.makedirs(TARIFFS_FOLDER, exist_ok=True)
os.makedirs(PROMPT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Variables pour stocker les noms des fichiers uploadés (simplifié pour le MVP)
uploaded_tariffs_files = []
uploaded_prompt_metier_file = None

# --- Fonctions utilitaires (simulées pour l'IA et le PDF) ---

def simulate_ai_response(user_message, current_quote_data):
    """
    Simule une réponse de l'IA et la génération/mise à jour du devis.
    Dans une vraie application, cela appellerait le LLM avec le contexte RAG.
    """
    response_text = ""
    updated_quote = current_quote_data # Copie pour modification

    if "devis" in user_message.lower() or "prépare" in user_message.lower():
        response_text = "Très bien, je commence à préparer le devis. J'ai besoin de quelques détails..."
        # Exemple de génération de devis initial
        updated_quote = {
            "entreprise": "Mon Entreprise Artisanale",
            "client": "M. Dupont",
            "adresse_client": "123 Rue de l'Exemple, 75001 Paris",
            "date": "2023-10-27",
            "lignes": [
                {"description": "Dépôt ancienne baignoire", "quantite": 1, "unite": "forfait", "prix_unitaire": 150.00, "total_ht": 150.00},
                {"description": "Fourniture et pose douche à l'italienne modèle X", "quantite": 1, "unite": "forfait", "prix_unitaire": 1200.00, "total_ht": 1200.00},
                {"description": "Pose faïence modèle Y", "quantite": 25, "unite": "m2", "prix_unitaire": 45.00, "total_ht": 1125.00},
                {"description": "Peinture plafond SDB", "quantite": 6, "unite": "m2", "prix_unitaire": 25.00, "total_ht": 150.00},
            ],
            "total_ht": 2625.00,
            "tva_rate": 0.20,
            "total_tva": 525.00,
            "total_ttc": 3150.00,
            "mentions": "Validité de l'offre : 30 jours. Conditions de règlement : 30% à la commande, solde à réception.",
            "prompt_metier_used": uploaded_prompt_metier_file
        }
    elif "change la quantité de faïence à" in user_message.lower():
        try:
            new_qty_str = user_message.split("faïence à")[1].strip().split("m2")[0].strip()
            new_qty = float(new_qty_str)
            for line in updated_quote.get("lignes", []):
                if "faïence" in line["description"].lower():
                    line["quantite"] = new_qty
                    line["total_ht"] = round(new_qty * line["prix_unitaire"], 2)
                    break
            # Recalculer les totaux
            updated_quote["total_ht"] = sum(line["total_ht"] for line in updated_quote["lignes"])
            updated_quote["total_tva"] = round(updated_quote["total_ht"] * updated_quote["tva_rate"], 2)
            updated_quote["total_ttc"] = round(updated_quote["total_ht"] + updated_quote["total_tva"], 2)
            response_text = f"J'ai mis à jour la quantité de faïence à {new_qty} m². Le devis est ajusté."
        except Exception:
            response_text = "Désolé, je n'ai pas compris la nouvelle quantité de faïence."
    else:
        response_text = "Je peux vous aider à générer un devis. Décrivez-moi simplement le chantier."

    return response_text, updated_quote

def generate_pdf_content(quote_data):
    """
    Simule la génération d'un contenu PDF.
    Dans une vraie application, cela utiliserait une bibliothèque PDF pour créer un vrai PDF formaté.
    """
    if not quote_data:
        return "Aucun devis à générer."

    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Devis - {quote_data.get('entreprise', 'Devis')}</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; font-size: 10pt; line-height: 1.4; }}
            .container {{ width: 800px; margin: auto; padding: 20px; border: 1px solid #eee; }}
            h1, h2 {{ color: #333; }}
            .header, .footer {{ text-align: center; margin-bottom: 20px; }}
            .client-info, .company-info {{ margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .total-section {{ text-align: right; }}
            .total-section div {{ margin-top: 5px; }}
            .mentions {{ font-size: 9pt; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="https://via.placeholder.com/100x50?text=Logo" alt="Logo Entreprise" style="float: left; margin-right: 20px;">
                <h1>Devis</h1>
                <p>Date : {quote_data.get('date', 'N/A')}</p>
            </div>
            <div style="clear: both;"></div>

            <div class="company-info">
                <h2>Votre Entreprise</h2>
                <p><strong>{quote_data.get('entreprise', 'Nom Entreprise')}</strong><br>
                Adresse: 1 Rue de l'Artisan, 75000 Paris<br>
                SIRET: 12345678900000<br>
                Assurance: AXA Pro (Contrat n°12345)</p>
            </div>

            <div class="client-info">
                <h2>Client</h2>
                <p><strong>{quote_data.get('client', 'Nom Client')}</strong><br>
                {quote_data.get('adresse_client', 'Adresse Client')}</p>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Description</th>
                        <th>Quantité</th>
                        <th>Unité</th>
                        <th>Prix Unitaire HT</th>
                        <th>Total HT</th>
                    </tr>
                </thead>
                <tbody>
    """
    for line in quote_data.get('lignes', []):
        html_content += f"""
                    <tr>
                        <td>{line.get('description', '')}</td>
                        <td>{line.get('quantite', '')}</td>
                        <td>{line.get('unite', '')}</td>
                        <td>{line.get('prix_unitaire', ''):.2f} €</td>
                        <td>{line.get('total_ht', ''):.2f} €</td>
                    </tr>
        """
    html_content += f"""
                </tbody>
            </table>

            <div class="total-section">
                <div>Total HT : <strong>{quote_data.get('total_ht', 0.00):.2f} €</strong></div>
                <div>TVA ({quote_data.get('tva_rate', 0.00)*100:.0f}%) : <strong>{quote_data.get('total_tva', 0.00):.2f} €</strong></div>
                <div>Total TTC : <strong>{quote_data.get('total_ttc', 0.00):.2f} €</strong></div>
            </div>

            <div class="mentions">
                <h3>Mentions Légales & Conditions</h3>
                <p>{quote_data.get('mentions', 'Aucune mention spécifique.')}</p>
                <p>Source Prompt Métier (simulé) : {quote_data.get('prompt_metier_used', 'N/A')}</p>
            </div>

            <div class="footer" style="margin-top: 50px;">
                <p>Merci de votre confiance !</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


# --- Routes Flask ---

@app.route('/')
def index():
    """Rend la page principale de l'application."""
    return render_template('index.html',
                           uploaded_tariffs=uploaded_tariffs_files,
                           uploaded_prompt=uploaded_prompt_metier_file)

@app.route('/upload_tariffs', methods=['POST'])
def upload_tariffs():
    """Gère l'upload des fichiers de tarifs."""
    if 'tariffsFiles' not in request.files:
        return jsonify({"success": False, "message": "Aucun fichier sélectionné"}), 400
    
    files = request.files.getlist('tariffsFiles')
    
    new_files = []
    for file in files:
        if file.filename == '':
            continue
        if file and file.filename.endswith('.pdf'): # Vérification simple du type
            filename = secure_filename(file.filename)
            file_path = os.path.join(TARIFFS_FOLDER, filename)
            file.save(file_path)
            uploaded_tariffs_files.append(filename) # Ajoute à la liste des fichiers en mémoire
            new_files.append(filename)
        else:
            return jsonify({"success": False, "message": f"Seuls les fichiers PDF sont acceptés. '{file.filename}' ignoré."}), 400
            
    return jsonify({"success": True, "message": f"{len(new_files)} fichiers de tarifs uploadés avec succès.", "files": uploaded_tariffs_files}), 200

@app.route('/upload_prompt_metier', methods=['POST'])
def upload_prompt_metier():
    """Gère l'upload du fichier 'Prompt Métier'."""
    global uploaded_prompt_metier_file # Nécessaire pour modifier la variable globale

    if 'promptMetierFile' not in request.files:
        return jsonify({"success": False, "message": "Aucun fichier sélectionné"}), 400
    
    file = request.files['promptMetierFile']
    if file.filename == '':
        return jsonify({"success": False, "message": "Aucun fichier sélectionné"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(PROMPT_FOLDER, filename)
        file.save(file_path)
        uploaded_prompt_metier_file = filename # Met à jour le nom du fichier en mémoire
        return jsonify({"success": True, "message": f"Fichier Prompt Métier '{filename}' uploadé avec succès.", "file": uploaded_prompt_metier_file}), 200
    else:
        return jsonify({"success": False, "message": "Seuls les fichiers PDF sont acceptés."}), 400

# Variable globale pour stocker l'état actuel du devis
current_quote = {}

@app.route('/chat', methods=['POST'])
def chat():
    """
    Gère les interactions de chat avec l'IA.
    Simule la logique de l'IA et la mise à jour du devis.
    """
    global current_quote # Nécessaire pour modifier la variable globale
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({"ai_message": "Veuillez taper un message.", "quote_preview": current_quote})

    # Simule la réponse de l'IA et la mise à jour du devis
    ai_response_text, updated_quote_data = simulate_ai_response(user_message, current_quote)
    current_quote = updated_quote_data # Met à jour le devis global

    return jsonify({"ai_message": ai_response_text, "quote_preview": current_quote})


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """Simule le téléchargement du devis au format PDF."""
    if not current_quote:
        return "Aucun devis à télécharger.", 404

    # Pour le MVP, nous générons un fichier HTML et le faisons passer pour un PDF.
    # Dans une vraie application, vous généreriez un vrai PDF.
    pdf_content = generate_pdf_content(current_quote)
    
    # Créer un fichier temporaire en mémoire pour l'envoyer
    buffer = io.BytesIO(pdf_content.encode('utf-8'))
    buffer.seek(0)

    # Envoyer le fichier HTML avec un header de type PDF
    # Le navigateur le téléchargera comme .pdf mais son contenu sera HTML
    return send_file(
        buffer,
        as_attachment=True,
        download_name="Devis_Artisan.html", # Ou .pdf si vous générez un vrai PDF
        mimetype="text/html" # Ou "application/pdf" pour un vrai PDF
    )


if __name__ == '__main__':
    app.run(debug=True)