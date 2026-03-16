import pandas as pd
import os
import re

CSV_INPUT = "3_results/nenets_whisper_ru_transcriptions.csv"
FINAL_CSV_PATH = "3_results/nenets_whisper_ru_for_review.csv"
TXT_REVIEW_PATH = "3_results/nenets_whisper_ru_for_review.txt"

print(f"Loading {CSV_INPUT}...")
df = pd.read_csv(CSV_INPUT, sep=';')

def extract_source(file_name):
    return re.sub(r'_seg\d+\.wav$', '', str(file_name))

df['source_file'] = df['file_name'].apply(extract_source)
df_ordered = df.sort_values(by=['source_file', 'file_name'])

# Save as ordered CSV
df_ordered.to_csv(FINAL_CSV_PATH, index=False, sep=';')
print(f"Saved ordered CSV to {FINAL_CSV_PATH}")

# Save as a readable text file
with open(TXT_REVIEW_PATH, 'w', encoding='utf-8') as f:
    current_source = ""
    for _, row in df_ordered.iterrows():
        source = row['source_file']
        if source != current_source:
            if current_source != "":
                f.write("\n")
            f.write(f"========================================\n")
            f.write(f"Source: {source}\n")
            f.write(f"========================================\n")
            current_source = source
        
        pred = str(row['prediction']) if pd.notna(row['prediction']) else ""
        f.write(f"[{row['file_name']}] {pred}\n")

print(f"Saved readable text file to {TXT_REVIEW_PATH}")
