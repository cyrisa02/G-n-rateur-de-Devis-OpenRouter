// ... (code JavaScript précédent) ...

function renderQuotePreview(quoteData) {
    const quotePreviewDiv = document.getElementById('quotePreview');
    if (Object.keys(quoteData).length === 0) {
        quotePreviewDiv.innerHTML = '<p>Le devis apparaîtra ici une fois que vous aurez commencé la conversation avec l\'IA.</p>';
        return;
    }

    let htmlContent = `
        <div class="quote-header">
            <h3>Devis pour ${quoteData.client || 'Client Inconnu'}</h3>
            <p><strong>Entreprise:</strong> ${quoteData.entreprise || 'Non spécifié'}</p>
            <p><strong>Adresse Client:</strong> ${quoteData.adresse_client || 'Non spécifié'}</p>
            <p><strong>Date:</strong> ${quoteData.date || 'Non spécifié'}</p>
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
    `;

    quoteData.lignes.forEach(line => {
        htmlContent += `
                <tr>
                    <td>${line.description}</td>
                    <td>${line.quantite}</td>
                    <td>${line.unite}</td>
                    <td>${line.prix_unitaire.toFixed(2)} €</td>
                    <td>${line.total_ht.toFixed(2)} €</td>
                </tr>
        `;
    });

    htmlContent += `
            </tbody>
        </table>
        <div class="total-section">
            <div>Total HT : <strong>${quoteData.total_ht.toFixed(2)} €</strong></div>
            <div>TVA (${(quoteData.tva_rate * 100).toFixed(0)}%) : <strong>${quoteData.total_tva.toFixed(2)} €</strong></div>
            <div>Total TTC : <strong>${quoteData.total_ttc.toFixed(2)} €</strong></div>
        </div>
        <div class="mentions">
            <h3>Mentions Légales & Conditions</h3>
            <p>${quoteData.mentions || 'Aucune mention spécifique.'}</p>
        </div>
    `;
    quotePreviewDiv.innerHTML = htmlContent;
}

function updateDownloadButtonState(hasQuote) {
    const downloadBtn = document.getElementById('downloadPdfBtn');
    if (hasQuote) {
        downloadBtn.removeAttribute('disabled');
    } else {
        downloadBtn.setAttribute('disabled', 'true');
    }
}

async function downloadPdf() {
    try {
        const response = await fetch('/download_pdf', {
            method: 'GET',
        });

        if (response.ok) {
            // Le backend envoie un fichier HTML pour simuler le PDF
            // Nous créons un lien temporaire pour télécharger ce fichier
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'Devis_Artisan.html'; // Le nom du fichier sera .html
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            alert("Erreur lors du téléchargement du devis. " + response.statusText);
        }
    } catch (error) {
        console.error('Erreur lors du téléchargement du PDF:', error);
        alert("Une erreur est survenue lors du téléchargement du devis.");
    }
}

// Envoyer le message si l'utilisateur appuie sur "Entrée" dans le champ de texte
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});