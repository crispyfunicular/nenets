# Récapitulatif — ASR Nénètse : Modèles et Résultats

**Corpus historique** : ~15 minutes (174 segments)  
**Ajout 2026-10-02** : 81 segments `KhO_ArcticReindeer` (~2,5 min) en train uniquement  
**Évaluation** : split 90/10 sur les 174 segments, `seed=42` (18 test) — protocole inchangé  
**Date** : 2026-10-02

## Tableau comparatif

| Modèle | Type | Fine-tuning | Pré-entraîné sur | WER | CER |
|--------|------|-------------|-------------------|-----|-----|
| **Wav2Vec2 XLSR** | CTC | gold + KhO (2 oct 2026) | Multilingue (XLSR-53) | **68.42%** | **15.51%** |
| Wav2Vec2 XLSR (avant) | CTC | Corpus 174 seg. | Multilingue (XLSR-53) | 70.87% | 16.47% |
| Whisper Small (russe) | Seq2seq | gold + KhO (2 oct 2026) | Multilingue, tokenizer russe | 106.32% | 57.25% |
| Whisper Small (russe, avant) | Seq2seq | Corpus 174 seg. | Multilingue, tokenizer russe | 72.82% | 22.84% |
| Whisper Small (sans langue) | Seq2seq | Corpus 174 seg. | Multilingue | 174.76% | 102.39% |
| Whisper Large v3 (russe) | Seq2seq | Corpus 174 seg. | Multilingue, tokenizer russe | 88.42% | 26.38% |
| **Omnilingual ZS** (7B) | Zero-shot | ❌ Aucun | Multilingue (1600+ langues) | 142.86% | 64.34% |

> Les scores « avant » / Omnilingual / Whisper Large / sans langue ne sont pas re-mesurés ici. Les lignes « 2 oct 2026 » viennent de `scripts/8_evaluate_cer_ru.py` (voir `eval_plus_2oct26_summary.md`).

## Notes sur les modèles

### Re-entraînés le 2026-10-02

- **Wav2Vec2 XLSR** — Toujours le meilleur modèle. Gain modeste sur le test fixe (−2,45 pts WER, −0,96 pts CER). Checkpoint : `2_models/wav2vec2-large-xlsr-nenets/`.
- **Whisper Small (russe)** — Se dégrade fortement sur le test historique (WER 106 %). Sur la val interne du dossier `plus_2oct26`, le WER d’entraînement tombait ~53 % : signe d’un mauvais transfert hors de ce split / overfit. Checkpoint : `2_models/whisper-small-nenets-ru/`.

### Non re-entraînés

- **Whisper Small (sans langue)** — Inutilisable (hallucinations).
- **Whisper Large v3 (russe)** — Moins bon que le Small historique avec peu de données.
- **Omnilingual ASR 7B** — Zero-shot, sans fine-tuning.

## Conclusion

> **XLSR reste le meilleur système** après l’ajout des données du 2 octobre (WER 68.42%, CER 15.51%).
>
> L’ajout de ~2,5 min de parole aide le CTC XLSR, mais **ne suffit pas** (et nuit) à Whisper Small RU sur le test fixe comparable.
>
> Prochaine étape utile : inspecter les erreurs Whisper (orthographe KhO / overfit) ou enrichir encore le corpus avant un nouveau fine-tuning Whisper.
