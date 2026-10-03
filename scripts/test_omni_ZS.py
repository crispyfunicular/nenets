"""
Inférence zero-shot avec Omnilingual ASR (omniASR_LLM_7B_ZS) pour le nénètse.

Utilise des segments transcrits comme exemples de contexte, puis transcrit
les segments non transcrits via l'API zero-shot d'Omnilingual.

Usage:
    python scripts/test_omni_ZS.py [--num-context N] [--batch-size B] [--output FILE]
"""

import csv
import os
import re
import argparse

from omnilingual_asr.models.inference.pipeline import (
    ASRInferencePipeline,
    ContextExample
)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Segments transcrits (contexte)
CONTEXT_AUDIO_DIR = os.path.join(BASE_DIR, "1_data_prepared", "processed_audio_16k")
CONTEXT_METADATA = os.path.join(CONTEXT_AUDIO_DIR, "metadata.csv")

# Segments à transcrire
INFERENCE_AUDIO_DIR = os.path.join(BASE_DIR, "1_data_prepared", "inference_segments")
INFERENCE_METADATA = os.path.join(INFERENCE_AUDIO_DIR, "metadata_inference.csv")

# Sortie
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "3_results", "omnilingual_zs_results.csv")

# Marqueurs à exclure des exemples de contexte (segments "sales")
DIRTY_MARKERS = re.compile(r"<\?\?\?>|<pause>|<a_d>|<p_r>|<u_l>|<e_r>|<f_s>|<c_l>|<m_l>|<u_w>|<ahh>|<rus>")


def load_context_examples(metadata_path, audio_dir, max_examples=10):
    """
    Charge des segments transcrits propres depuis metadata.csv pour servir
    d'exemples de contexte zero-shot.

    Filtre les transcriptions contenant des marqueurs de bruit/incertitude
    et privilégie les segments de durée moyenne (2–8 secondes).
    """
    candidates = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcription = row["transcription"].strip()
            duration = float(row["duration"])
            filename = row["file_name"]
            audio_path = os.path.join(audio_dir, filename)

            # Filtrer les segments sales
            if DIRTY_MARKERS.search(transcription):
                continue

            # Filtrer les segments trop courts ou trop longs
            if duration < 2.0 or duration > 8.0:
                continue

            # Vérifier que le fichier audio existe
            if not os.path.exists(audio_path):
                continue

            candidates.append({
                "path": audio_path,
                "transcription": transcription,
                "duration": duration,
            })

    # Trier par durée (segments moyens d'abord, ~4s idéal)
    candidates.sort(key=lambda x: abs(x["duration"] - 4.0))

    selected = candidates[:max_examples]
    print(f"Exemples de contexte sélectionnés : {len(selected)}/{len(candidates)} candidats")
    for i, ex in enumerate(selected):
        print(f"  [{i+1}] {os.path.basename(ex['path'])} ({ex['duration']:.1f}s) : {ex['transcription'][:60]}...")

    return [ContextExample(ex["path"], ex["transcription"]) for ex in selected]


def load_inference_files(metadata_path, audio_dir):
    """
    Charge la liste des fichiers à transcrire depuis metadata_inference.csv.
    Retourne une liste de (filename, audio_path).
    """
    files = []
    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["file_name"].strip()
            audio_path = os.path.join(audio_dir, filename)
            if os.path.exists(audio_path):
                files.append((filename, audio_path))
            else:
                print(f"  ATTENTION : fichier manquant {audio_path}")
    print(f"Fichiers à transcrire : {len(files)}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Inférence zero-shot Omnilingual ASR pour le nénètse"
    )
    parser.add_argument(
        "--num-context", type=int, default=10,
        help="Nombre d'exemples de contexte (défaut: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Taille des lots pour l'inférence (défaut: 1)"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Fichier de sortie CSV (défaut: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    # --- 1. Charger le modèle ---
    print("=" * 60)
    print("OMNILINGUAL ASR — INFÉRENCE ZERO-SHOT (NÉNÈTSE)")
    print("=" * 60)
    print(f"\nChargement du modèle omniASR_LLM_7B_ZS...")
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS")
    print("Modèle chargé.\n")

    # --- 2. Charger les exemples de contexte ---
    print("--- Sélection des exemples de contexte ---")
    context_examples = load_context_examples(
        CONTEXT_METADATA, CONTEXT_AUDIO_DIR, max_examples=args.num_context
    )
    if not context_examples:
        print("ERREUR : aucun exemple de contexte trouvé !")
        return

    # --- 3. Charger les fichiers à transcrire ---
    print("\n--- Chargement des fichiers à transcrire ---")
    inference_files = load_inference_files(INFERENCE_METADATA, INFERENCE_AUDIO_DIR)
    if not inference_files:
        print("ERREUR : aucun fichier à transcrire trouvé !")
        return

    # --- 4. Transcrire par lots ---
    print(f"\n--- Début de l'inférence (batch_size={args.batch_size}) ---")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    total = len(inference_files)

    for i in range(0, total, args.batch_size):
        batch = inference_files[i:i + args.batch_size]
        batch_paths = [path for _, path in batch]
        batch_names = [name for name, _ in batch]

        # Chaque fichier du batch utilise les mêmes exemples de contexte
        batch_context = [context_examples] * len(batch_paths)

        transcriptions = pipeline.transcribe_with_context(
            batch_paths,
            context_examples=batch_context,
            batch_size=args.batch_size
        )

        for name, transcription in zip(batch_names, transcriptions):
            results.append({"file_name": name, "transcription": transcription})

        done = min(i + args.batch_size, total)
        print(f"  Progression : {done}/{total} ({100*done/total:.0f}%)")

    # --- 5. Sauvegarder les résultats ---
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"Terminé ! {len(results)} transcriptions sauvegardées dans :")
    print(f"  {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
