import pandas as pd
import os
import re

# 1. Configuration des dossiers
# On s'adapte à ta structure : les TextGrids vont dans 'inference_textgrids'
CSV_INPUT = "3_results/nenets_unlabeled_transcriptions.csv"
TG_DIR = "1_data_prepared/inference_textgrids/"
FINAL_CSV_PATH = "3_results/metadata_inference_ordered.csv"

# Créer le dossier TextGrid s'il n'existe pas encore
if not os.path.exists(TG_DIR):
    os.makedirs(TG_DIR)

# 2. Chargement et tri des données
df = pd.read_csv(CSV_INPUT, sep=';')

# Extraction du nom du fichier source (on enlève '_segXXX.wav')
def extract_source(file_name):
    return re.sub(r'_seg\d+\.wav$', '', file_name)

df['source_file'] = df['file_name'].apply(extract_source)

# Tri : d'abord par texte source, puis par ordre chronologique des segments
df_ordered = df.sort_values(by=['source_file', 'file_name'])

# Sauvegarde du CSV "propre" pour Nikolett
df_ordered.to_csv(FINAL_CSV_PATH, index=False, sep=';')

# 3. Génération des TextGrids avec le texte prédit
for _, row in df_ordered.iterrows():
    file_name = row['file_name']
    # On récupère la prédiction (si c'est <>, on laisse tel quel)
    prediction = str(row['prediction']) if pd.notna(row['prediction']) else ""
    
    # Nettoyage des guillemets pour ne pas casser le format TextGrid
    prediction = prediction.replace('"', "'")
    
    tg_name = file_name.replace('.wav', '.TextGrid')
    tg_path = os.path.join(TG_DIR, tg_name)
    
    # Structure ooTextFile pour Praat
    # On met xmax=100 par défaut car Praat ajustera à la durée réelle du .wav
    content = f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 100 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "transcription_auto" 
        xmin = 0 
        xmax = 100 
        intervals: size = 1 
        intervals [1]:
            xmin = 0 
            xmax = 100 
            text = "{prediction}" 
"""
    with open(tg_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"Succès : {len(df_ordered)} TextGrids générés dans {TG_DIR}")
print(f"CSV ordonné créé : {FINAL_CSV_PATH}")