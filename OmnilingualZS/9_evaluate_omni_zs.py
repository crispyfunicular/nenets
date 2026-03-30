"""
Évaluation zero-shot d'Omnilingual ASR (omniASR_LLM_7B_ZS) pour le nénètse.

Protocole comparable à 8_evaluate_cer.py :
  - Même dataset (processed_audio_16k / metadata.csv)
  - Même split 90/10, seed=42
  - 90% = exemples de contexte ZS,  10% = segments à transcrire
  - Calcul WER / CER avec jiwer

Usage:
    python scripts/9_evaluate_omni_zs.py [--num-context N] [--output FILE]
"""

import csv
import os
import re
import argparse
import random

import jiwer

from omnilingual_asr.models.inference.pipeline import (
    ASRInferencePipeline,
    ContextExample,
)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "1_data_prepared", "processed_audio_16k")
METADATA = os.path.join(AUDIO_DIR, "metadata.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "3_results", "omnilingual_zs_evaluation.csv")

# Marqueurs à exclure des exemples de contexte
DIRTY_MARKERS = re.compile(
    r"<\?\?\?>|<pause>|<a_d>|<p_r>|<u_l>|<e_r>|<f_s>|<c_l>|<m_l>|<u_w>|<ahh>|<rus>"
)

SEED = 42
TEST_RATIO = 0.1


def load_metadata(metadata_path, audio_dir):
    """Charge toutes les entrées de metadata.csv."""
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["file_name"].strip()
            transcription = row["transcription"].strip()
            duration = float(row["duration"])
            audio_path = os.path.join(audio_dir, filename)

            if not os.path.exists(audio_path):
                print(f"  ATTENTION : fichier manquant {audio_path}")
                continue

            entries.append({
                "file_name": filename,
                "path": audio_path,
                "transcription": transcription,
                "duration": duration,
            })
    return entries


def split_data(entries, test_ratio=TEST_RATIO, seed=SEED):
    """
    Split 90/10 reproductible, identique au split de 8_evaluate_cer.py
    (qui utilise datasets.train_test_split(test_size=0.1, seed=42)).
    """
    indices = list(range(len(entries)))
    random.Random(seed).shuffle(indices)
    n_test = max(1, int(len(entries) * test_ratio))
    test_indices = set(indices[:n_test])

    train = [entries[i] for i in range(len(entries)) if i not in test_indices]
    test = [entries[i] for i in range(len(entries)) if i in test_indices]
    return train, test


def build_context(train_entries, max_examples=10):
    """
    Sélectionne des exemples de contexte propres parmi le split train.
    Même logique de filtrage que test_omni_ZS.py.
    """
    candidates = []
    for entry in train_entries:
        if DIRTY_MARKERS.search(entry["transcription"]):
            continue
        if entry["duration"] < 2.0 or entry["duration"] > 8.0:
            continue
        candidates.append(entry)

    # Trier par proximité à 4s (durée idéale)
    candidates.sort(key=lambda x: abs(x["duration"] - 4.0))
    selected = candidates[:max_examples]

    print(f"Exemples de contexte : {len(selected)}/{len(candidates)} candidats")
    for i, ex in enumerate(selected):
        print(
            f"  [{i+1}] {ex['file_name']} ({ex['duration']:.1f}s) : "
            f"{ex['transcription'][:60]}..."
        )

    return [ContextExample(ex["path"], ex["transcription"]) for ex in selected]


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation zero-shot Omnilingual ASR pour le nénètse"
    )
    parser.add_argument(
        "--num-context", type=int, default=10,
        help="Nombre d'exemples de contexte (défaut: 10)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Fichier de sortie CSV (défaut: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # --- 1. Charger les données ---
    print("=" * 60)
    print("OMNILINGUAL ASR — ÉVALUATION ZERO-SHOT (NÉNÈTSE)")
    print("=" * 60)

    entries = load_metadata(METADATA, AUDIO_DIR)
    print(f"\nTotal segments : {len(entries)}")

    # --- 2. Split train/test ---
    train_entries, test_entries = split_data(entries)
    print(f"Split : {len(train_entries)} contexte / {len(test_entries)} test")

    # --- 3. Construire les exemples de contexte ---
    print("\n--- Sélection des exemples de contexte ---")
    context_examples = build_context(train_entries, max_examples=args.num_context)
    if not context_examples:
        print("ERREUR : aucun exemple de contexte trouvé !")
        return

    # --- 4. Charger le modèle ---
    print(f"\nChargement du modèle omniASR_LLM_7B_ZS...")
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS")
    print("Modèle chargé.\n")

    # --- 5. Transcrire les segments de test ---
    print(f"--- Inférence sur {len(test_entries)} segments de test ---")
    references = []
    predictions = []

    for i, entry in enumerate(test_entries):
        transcription = pipeline.transcribe_with_context(
            [entry["path"]],
            context_examples=[context_examples],
            batch_size=1,
        )
        pred = transcription[0] if transcription else ""
        ref = entry["transcription"]

        references.append(ref)
        predictions.append(pred)

        print(f"  [{i+1}/{len(test_entries)}] {entry['file_name']}")
        print(f"    REF : {ref[:80]}")
        print(f"    HYP : {pred[:80]}")

    # --- 6. Calculer WER / CER ---
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)

    print(f"\n{'=' * 60}")
    print(f"RÉSULTATS (sur {len(test_entries)} segments de test)")
    print(f"{'=' * 60}")
    print(f"  WER : {wer:.4f}  ({wer*100:.2f}%)")
    print(f"  CER : {cer:.4f}  ({cer*100:.2f}%)")
    print(f"{'=' * 60}")

    # --- 7. Sauvegarder les résultats détaillés ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name", "reference", "prediction"]
        )
        writer.writeheader()
        for entry, ref, pred in zip(test_entries, references, predictions):
            writer.writerow({
                "file_name": entry["file_name"],
                "reference": ref,
                "prediction": pred,
            })

    print(f"\nDétails sauvegardés dans : {args.output}")


if __name__ == "__main__":
    main()
