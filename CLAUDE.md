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

## Clustering Results (Goal 4)

K-means on full 429-feature matrix (1110 sequences: 500 bio + 500 scrambled + 110 designed).
Elbow method selected k=7. PCA top 2 PCs explain 8.0% of variance (4.6% + 3.4%).

| Cluster | Biological | Scrambled | Designed | Notes                                         |
|---------|-----------|-----------|----------|-----------------------------------------------|
| 0       | 9         | 7         | 68       | TolA_III, BindCraft (lpg0945), RAVj, lpg0944 |
| 1       | 171       | 170       | 0        | Pure biological/scrambled zone               |
| 2       | 1         | 1         | 19       | rank_design_* sequences (distinct program)   |
| 3       | 6         | 0         | 0        | Small biological-only cluster                |
| 4       | 143       | 146       | 11       | Mostly bio/scrambled; some latent binders    |
| 5       | 35        | 36        | 4        | Mostly bio/scrambled; some latent binders    |
| 6       | 135       | 140       | 8        | Mostly bio/scrambled; some latent binders    |

Key findings:
- 79% of designed binders (87/110) land in clusters 0 and 2 — distinct from biological space.
- Design program signal: TolA_III/BindCraft/RAVj designs → cluster 0; rank_design → cluster 2.
- `latent` and `0945_latent` sequences scatter into biological clusters (4–6), suggesting
  those designs look more realistic at the sequence composition level.
- Biological sequences expected to form multiple sub-clusters (membrane/soluble/disordered)
  confirmed by k=7 rather than k=3 being the natural grouping.

Output files: results/figures/clustering_elbow.png, results/figures/clustering_pca.png,
results/clustering_run.txt

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

### 4. Unsupervised clustering
Scientific question (from Dr. Godzik): do the 110 designed TolA binders cluster with
soluble proteins (expected, since TolA is periplasmic), or do some look non-biological?
Which design programs produce sequences that look most realistic?

Key insight: "Biological" is not one group. Membrane, soluble, and disordered/low-complexity
proteins differ systematically in AA composition and dipeptide frequencies. Do not pre-assume
k=3 — let the data reveal natural groupings.

Implementation:
- Run K-means on the full 429-feature matrix (all 1110 sequences: bio + scrambled + designed)
- Use the elbow method to determine k; expect biological sequences to form multiple sub-clusters
  (membrane / soluble / disordered) rather than one monolithic group
- PCA for initial 2D visualization (UMAP optionally after)
- Color points by cluster assignment; mark designed binders with a distinct marker shape
  (e.g. star or triangle) so cluster membership and sequence origin are both visible
- Annotate or flag which design programs produced binders that fall into soluble/biological clusters
- Save plot to results/figures/clustering_pca.png
- This is the scientific deliverable for Dr. Godzik — treat it as the main output

---

## Known Issues / Refactor TODOs

1. **No `if __name__ == "__main__":` guard** — argparse and all execution code run at module
   level. Goal 4 (clustering) will need to import feature-building functions (e.g.
   `build_feature_vector`, `get_feature_names`) from this file; without the guard, any
   import triggers the full training pipeline. Fix this before adding clustering as a
   separate script or module.

2. **`print_random_forest_importance` is reused for XGBoost** (Step 4 in the pipeline).
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
- Add `if __name__ == "__main__":` guard when refactoring
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