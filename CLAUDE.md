# Protein Sequence Classifier
## Godzik Lab · UCR School of Medicine

Supervised ML classifier distinguishing biological, scrambled, and AI-designed protein sequences.
Scientific goal: understand what makes AI-designed TolA binders different from natural protein sequence space.

---

## Code State

Main script: `src/train.py`
Run command: `python src/train.py --fasta /Users/samsharma/Projects/protein-sequence-classifier/data/raw/uniprotkb_reviewed_true_AND_reviewed_tr_2026_02_26.fasta.gz --binders /Users/samsharma/Projects/protein-sequence-classifier/data/raw/Binders.fasta`

The script is a single-file pipeline with no `if __name__ == "__main__":` guard — 
the argparse and execution code runs at module level. Fix this during refactor.

---

## Feature Engineering (429 features — do not change without explicit instruction)

| Feature Group         | Count | Description                                                   |
|-----------------------|-------|---------------------------------------------------------------|
| Amino acid composition| 20    | Per-residue frequency of each of the 20 standard AAs         |
| Dipeptide composition | 400   | Normalized frequency of all AA pairs (ordered)               |
| Physicochemical       | 9     | MW/residue, aromaticity, instability index, pI, GRAVY,        |
|                       |       | helix/turn/sheet fractions (secondary structure), net charge  |

Feature order is fixed by `get_feature_names()`. The 429-feature design is intentional — 
results depend on this exact set. Do not add or remove features without explicit instruction.

---

## Models

Two models currently implemented:
- `LogisticRegression` — trained on StandardScaler-normalized features, max_iter=2000, class_weight='balanced'
- `RandomForestClassifier` — trained on raw (unscaled) features, n_estimators=200

Split: 80/20, stratified by class label, random_state=42.

Classes:
- Class 0: Biological (real UniProt Swiss-Prot sequences, loaded from gzipped FASTA)
- Class 1: Scrambled (same sequences, randomly shuffled — same composition, destroyed order)
- Class 2: Designed (TolA binder FASTA from Teresa He, Godzik Lab)

---

## Current Results

3-class mode (Biological / Scrambled / Designed), 500 bio + 500 scrambled + 110 designed,
80/20 split, random_state=42. Scrambling seeded with random.seed(42) — results are now
reproducible run-to-run.

| Model                 | Accuracy | F1 macro | F1 (Designed) | Notes                         |
|-----------------------|----------|----------|---------------|-------------------------------|
| LogisticRegression    | 67.12%   | 0.67     | 0.68          | Scaled features               |
| RandomForestClassifier| 66.67%   | 0.74     | 0.95          | Unscaled features             |
| XGBClassifier         | 69.82%   | 0.77     | 0.98          | Unscaled features; best overall|

Key findings:
- XGBoost is the best overall model (highest accuracy, F1 macro 0.77, F1 Designed 0.98).
- Both tree-based models dramatically outperform LR at identifying Designed binders
  (F1 0.95–0.98 vs 0.68). LR confuses most binders with Biological or Scrambled sequences.
- For the scientific goal (identifying what makes AI-designed binders distinct),
  XGBoost is the preferred model.

Validation: single 80/20 train/test split. Not cross-validated yet.

---

## Clustering Results (Goal 4) — COMPLETE

Two analyses implemented in `src/cluster.py`. Run command:
```
python src/cluster.py --fasta data/raw/...fasta.gz --binders data/raw/Binders.fasta
```

### Analysis 1: Full-matrix clustering (bio + scrambled + designed, 1110 sequences)
K-means on 429-feature matrix. Elbow → k=7. PCA top 2 PCs: 4.6% + 3.4% = 8.0%.

| Cluster | Bio | Scrambled | Designed | Character                                      |
|---------|-----|-----------|----------|------------------------------------------------|
| 0       | 9   | 7         | 68       | TolA_III, BindCraft (lpg0945), RAVj, lpg0944  |
| 1       | 171 | 170       | 0        | Pure biological/scrambled zone                 |
| 2       | 1   | 1         | 19       | rank_design_* sequences (distinct program)     |
| 3       | 6   | 0         | 0        | Small biological-only cluster                  |
| 4–6     | ~140| ~140      | 8–11     | Mostly bio/scrambled; latent binders mixed in  |

Key findings:
- 79% of designed binders (87/110) in clusters 0 + 2 — clearly distinct from biological space.
- Design programs separate cleanly: TolA_III/BindCraft/RAVj → cluster 0; rank_design → cluster 2.
- `latent` and `0945_latent` sequences scatter into biological clusters (4–6) — most realistic.

### Analysis 2: Bio-only clustering + designed binder projection
K-means on 500 biological sequences only. Elbow → k=7. PCA fit on bio; designed binders
transformed in. Clusters labeled by mean GRAVY, instability index, net charge.

| Cluster | N   | GRAVY  | Instability | Label      |
|---------|-----|--------|-------------|------------|
| 0       | 142 | -0.171 | 42.2        | Disordered |
| 1       | 7   | -0.601 | 34.9        | Soluble    |
| 2       | 91  | -0.615 | 53.4        | Disordered |
| 3       | 91  | -0.335 | 37.7        | Soluble    |
| 4       | 1   | -0.572 | 48.7        | Disordered |
| 5       | 141 | -0.113 | 34.8        | Soluble    |
| 6       | 27  | -0.267 | 45.0        | Disordered |

Designed binder projection (nearest bio cluster centroid in PCA space):
- Cluster 2 (Disordered): 43 binders — TolA_III hallucination designs
- Cluster 3 (Soluble):    33 binders — RAVj, lpg0944, some TolA_III
- Cluster 5 (Soluble):    17 binders — rank_design (most soluble-like)
- Cluster 6 (Disordered):  8 binders — mixed
- Remaining clusters:      9 binders

Design program summary:
- **rank_design** → Soluble (cluster 5) — closest to biological core, most realistic
- **latent / 0945_latent** → mostly Soluble — second-most realistic
- **TolA_III hallucination** → Disordered (cluster 2, instability 53.4)
- **RAVj / lpg0944** → split Soluble/Disordered

Known limitation: No membrane-like clusters emerged (all GRAVY < 0). The current
Swiss-Prot sample is likely soluble-biased. To properly test Dr. Godzik's membrane /
soluble / disordered hypothesis, a curated sample with representative membrane proteins
and IDPs is needed. This is the next data collection step.

Output files:
- results/figures/clustering_elbow.png, clustering_pca.png (full-matrix)
- results/figures/clustering_bio_only_elbow.png, clustering_bio_only_pca.png (bio-only)
- results/clustering_run.txt

---

## Data Files (not committed to git — large/proprietary)

- Biological sequences: UniProt Swiss-Prot reviewed, gzipped FASTA (~300MB), local path hardcoded in DEFAULT_FASTA_PATH
- Designed binders: 110 TolA binder sequences, non-gzipped FASTA from Teresa He
- Scrambled sequences: generated in-memory from biological sequences — no file needed

Data now lives in data/raw/. Always pass --fasta and --binders explicitly when running.
Do not commit data files.

---

## Immediate Goals (in order — do not skip ahead)

### 1. Add XGBoost as third model
- Same train/test split, same random_state, same features as existing models
- XGBoost takes raw features (like RandomForest, no scaling needed)
- Use `XGBClassifier` from the `xgboost` package

### 2. Clean model comparison table
- Print a single table at the end comparing all three models side by side
- Columns: Model, Accuracy, Precision (macro), Recall (macro), F1 (macro)
- Also per-class F1 for each model in the same table or a follow-up table
- Replace the current per-model printing with this unified output

### 3. Docstrings with biological reasoning
- Every function needs a docstring explaining not just what the code does but WHY 
  biologically (e.g., why dipeptide frequencies matter for classification)
- Update existing docstrings to include biological context

### 4. Unsupervised clustering ✓ COMPLETE
Implemented in `src/cluster.py`. See Clustering Results section above for full findings.

Remaining scientific gap: the Swiss-Prot sample is soluble-biased — no membrane clusters
emerged. Next step (not yet implemented) is to curate a balanced bio sample with membrane,
soluble, and IDP representatives before re-running the bio-only analysis.

---

## Known Issues / Refactor TODOs

1. **`print_random_forest_importance` is reused for XGBoost** (Step 4 in the pipeline).
   This works because both models expose `feature_importances_`, but the name is misleading
   at the call site. Rename to something model-agnostic (e.g. `print_tree_importance`)
   during the next refactor pass.

3. **`DEFAULT_FASTA_PATH` (line 30) is stale** — points to `~/Downloads/`, but data now
   lives in `data/raw/`. Not currently breaking anything since `--fasta` is always passed
   explicitly in the documented run command, but the default should be updated or removed
   during refactor to avoid confusion.

---

## Constraints

- Do not change feature count (429) or feature order without explicit instruction
- Keep --fasta and --binders CLI arguments working
- Keep 2-class (binary) and 3-class modes both functional
- Do not commit data files (.fasta, .fasta.gz, .csv of sequences)

---

## Stack

Python 3.10 · scikit-learn · BioPython · NumPy · XGBoost (install: pip install xgboost) · matplotlib · seaborn (for clustering plots)

---

## Git Conventions

Semantic commit prefix required on every commit:
- `feat:` new functionality
- `fix:` bug fix
- `refactor:` restructuring without behavior change
- `docs:` docstrings, README, CLAUDE.md
- `exp:` experiment run or results update
- `chore:` dependencies, config

End-of-session habit: `git add -A && git commit -m "type: description" && git push origin main`

--

## Output Convention
Save run outputs to results/ as descriptive .txt files (e.g. results/baseline_xgboost.txt).