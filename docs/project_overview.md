# Project overview

## Project

**Multimodal AI for Cause-of-Death Estimation from Postmortem Data** is a Group 2 capstone for CSE 499A in the Department of Electrical and Computer Engineering. The supervisor is Dr. Sumaiya Tabassum Nimi.

## Aim

Estimate broad cause-of-death categories from postmortem CT (PMCT) and non-leaky structured metadata from the New Mexico Decedent Image Database (NMDID), then measure what the images add beyond metadata.

The conceptual pipeline is foundation model → image embedding → fusion with reviewed metadata → prediction → evaluation. The current implemented and evaluated work is metadata-only. PMCT preprocessing, image embeddings, and multimodal fusion have not started, and no image or multimodal results exist.

## Current metadata cohort

The prepared cohort has 860 decedents and five labels:

| Cause category | Count |
|---|---:|
| Intoxication | 366 |
| Cardiac | 202 |
| Blunt trauma | 149 |
| Stab | 79 |
| Firearm | 64 |

The 70 numerical features are organized into six groups: basic metadata (5), diagnoses (22), medical history (5), substance use (10), lifestyle (12), and medications (16). Tier 1 contains the five basic features; Tier 1+2 contains all 70.

## Status

Metadata collection/preparation, metadata-only models, the TabPFN experiment, and the age ablation are complete. PMCT preprocessing is not started; PMCT embeddings and multimodal fusion are pending. Current metrics are preliminary repeated cross-validation results, with no held-out test set.
