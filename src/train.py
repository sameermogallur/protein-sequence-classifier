# train.py
# #Protein Classifier - Sameer Mogallur - Godzik Lab
#
# Classifies biological vs scrambled (and optionally designed binder) protein sequences
# using amino acid frequencies, dipeptide frequencies, and physicochemical features.
#
# Dependencies:
#   biopython>=1.79
#   scikit-learn>=1.0
#   numpy

import argparse
from collections import Counter
import gzip
import random

import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Constants
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_FASTA_PATH = "/Users/samsharma/Downloads/uniprotkb_reviewed_true_AND_reviewed_tr_2026_02_26.fasta.gz"

DIPEPTIDE_NAMES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
PHYSICOCHEMICAL_NAMES = [
    "per_residue_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy",
    "helix_fraction",
    "turn_fraction",
    "sheet_fraction",
    "net_charge_per_residue",
]


def get_feature_names():
    """Return the 429-element feature name list in fixed column order.

    The order (20 AA frequencies → 400 dipeptides → 9 physicochemical) is a contract:
    every model and every run must receive features in this exact column arrangement.
    Reordering silently breaks cross-run comparisons and any serialized model.
    """
    return list(AMINO_ACIDS) + DIPEPTIDE_NAMES + PHYSICOCHEMICAL_NAMES


def load_fasta(fasta_path, max_sequences=None, min_length=50):
    """Load protein sequences from a FASTA file (plain or gzip); return list of sequence strings.

    Swiss-Prot reviewed sequences are the gold standard for natural proteins — manually
    curated, non-redundant, and free of predicted or erroneous entries. The min_length=50
    filter drops fragments too short to yield reliable physicochemical signals (e.g.,
    isoelectric point and secondary structure estimates are unstable on very short peptides).
    """
    opener = gzip.open if fasta_path.endswith(".gz") else open
    mode = "rt" if fasta_path.endswith(".gz") else "r"
    sequences = []
    with opener(fasta_path, mode) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            if len(seq) >= min_length:
                sequences.append(seq)
                if max_sequences is not None and len(sequences) >= max_sequences:
                    break
    return sequences


def load_fasta_plain_with_ids(fasta_path, min_length=50):
    """Load sequence IDs and strings from a non-gzipped FASTA; return parallel lists.

    The designed binders (from Teresa He, Godzik Lab) are delivered as a plain FASTA,
    not Swiss-Prot. Sequence IDs are preserved alongside sequences so the per-binder
    prediction table can be traced back to individual designs — essential for deciding
    which candidates to advance to experimental validation.
    """
    ids = []
    sequences = []
    with open(fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            if len(seq) >= min_length:
                ids.append(record.id)
                sequences.append(seq)
    return ids, sequences


def scramble_sequence(seq):
    """Return a new string with the same amino acids in a uniformly random order.

    Scrambling is the null model for "does sequence order matter?" A scrambled sequence
    has identical amino acid composition to its biological source — same hydrophobicity,
    same charge, same monomer frequencies — but all order-dependent signals are destroyed:
    dipeptide preferences, secondary structure propensities, binding motifs, and folding
    nucleation sites. A classifier that successfully separates biological from scrambled
    sequences must be detecting order information, not just compositional bias.
    """
    return "".join(random.sample(seq, len(seq)))


def get_aa_frequencies(sequence):
    """Return a 20-element list of per-residue amino acid frequencies in AMINO_ACIDS order.

    Amino acid composition reflects evolutionary pressures and functional requirements.
    Natural proteins are constrained by cellular chemistry (e.g., Cys is rare and usually
    structural or redox-active). AI-designed binders optimized for a single target may be
    enriched in binding-favored residues — Trp, Arg, Tyr — at frequencies that deviate
    from the natural background. Scrambled sequences share the identical AA distribution as
    their biological parent by construction, so composition alone cannot separate bio from
    scrambled; it can, however, help distinguish designed from both.
    """
    length = len(sequence)
    counts = Counter(sequence)
    return [counts.get(aa, 0) / length for aa in AMINO_ACIDS]


def get_dipeptide_frequencies(sequence):
    """Return 400 normalized dipeptide frequencies over the 20-AA alphabet.

    Dipeptide frequencies capture the first layer of sequence order: which amino acids
    prefer to appear adjacent to which others. These preferences encode secondary structure
    propensities (e.g., Glu-Lys pairs favor alpha-helix; Pro disrupts it), coevolutionary
    constraints, and folding nucleation signals. Scrambled sequences preserve monomer
    composition but randomize all dipeptide statistics — making this feature group the
    primary discriminator between biological and scrambled sequences, and the most
    informative single block in the 429-feature vector for that classification task.
    """
    aa_set = set(AMINO_ACIDS)
    counts = Counter()
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i + 1]
        if a in aa_set and b in aa_set:
            counts[a + b] += 1
    total = sum(counts.values())
    if total == 0:
        return [0.0] * 400
    return [counts.get(d, 0) / total for d in DIPEPTIDE_NAMES]


def get_physicochemical_features(sequence):
    """Return 9 whole-sequence physicochemical properties; zeros if cleaned length < 10.

    These properties reflect chemistry that both evolution and computational design optimize,
    but for different objectives:
    - MW/residue: proxy for sequence complexity and average residue bulk.
    - Aromaticity: Phe/Trp/Tyr fraction; aromatic residues are rare in natural proteins but
      critical for hydrophobic core packing and binding hot-spots.
    - Instability index: >40 predicts in-vivo instability (Guruprasad et al.); designed
      binders are often specifically engineered for thermal stability.
    - pI: designed binders may carry atypical charge distributions tuned for electrostatic
      complementarity with the TolA binding surface.
    - GRAVY (Grand Average of Hydropathicity): separates membrane proteins (positive) from
      soluble globular proteins (negative); unfolded/scrambled sequences trend toward zero.
    - Helix/turn/sheet fractions: predicted from Chou-Fasman propensities; AI-designed
      proteins can adopt non-natural secondary structure distributions.
    - Net charge/residue: (K+R-D-E)/length; charge balance shapes solubility and
      complements the electrostatic surface of the binding partner.
    """
    valid = set(AMINO_ACIDS)
    cleaned = "".join(c for c in sequence if c in valid)
    n = len(cleaned)
    if n < 10:
        return [0.0] * 9
    pa = ProteinAnalysis(cleaned)
    mw_per_res = pa.molecular_weight() / n
    arom = pa.aromaticity()
    inst = pa.instability_index()
    p_i = pa.isoelectric_point()
    gv = pa.gravy()
    helix, turn, sheet = pa.secondary_structure_fraction()
    counts = Counter(cleaned)
    net_charge = (counts.get("K", 0) + counts.get("R", 0) - counts.get("D", 0) - counts.get("E", 0)) / n
    return [mw_per_res, arom, inst, p_i, gv, helix, turn, sheet, net_charge]


def build_feature_vector(sequence):
    """Concatenate all feature groups into the 429-element vector for one sequence.

    The column order — AA composition (20) → dipeptides (400) → physicochemical (9) —
    matches get_feature_names() and is the fixed input contract for all three models.
    Changing this order without retraining silently corrupts predictions because tree
    splits and linear coefficients are tied to specific column indices, not feature names.
    """
    return get_aa_frequencies(sequence) + get_dipeptide_frequencies(sequence) + get_physicochemical_features(sequence)


def compute_aa_freqs_from_sequences(sequences):
    """Compute background amino acid frequencies pooled across all sequences.

    The resulting distribution (computed from the biological Swiss-Prot subset) represents
    natural protein sequence space and serves as a reference against which designed and
    scrambled sequences can be compared. By construction, scrambled sequences will match
    this distribution exactly; deviations in the designed set reveal compositional biases
    introduced by the AI design process (e.g., over-representation of binding-hot-spot
    residues like Trp or Tyr).
    """
    counts = Counter()
    total = 0
    for seq in sequences:
        for aa in seq:
            if aa in AMINO_ACIDS:
                counts[aa] += 1
                total += 1
    if total == 0:
        raise ValueError("No valid amino acids found.")
    return {aa: counts.get(aa, 0) / total for aa in AMINO_ACIDS}


def print_confusion_matrix_block(cm, class_names):
    """Print a formatted confusion matrix for 2- or 3-class problems.

    Confusion matrices reveal which classes are being conflated, which matters scientifically.
    The critical error to watch is Designed predicted as Biological — it means the model
    cannot distinguish AI-designed binders from natural proteins, which undermines the core
    research question. Biological-as-Scrambled errors are less consequential scientifically.
    Always examine the Designed row (row 2) in 3-class mode.
    """
    n = len(class_names)
    if n == 2:
        print("                 Predicted Bio  Predicted Scr")
        print(f"  Actual Bio:     {cm[0][0]:>10}     {cm[0][1]:>10}")
        print(f"  Actual Scr:     {cm[1][0]:>10}     {cm[1][1]:>10}")
        return
    abbr = [name[:3] if len(name) > 3 else name for name in class_names]
    header = " " * 18 + "".join(f"  Pred {a:>3}" for a in abbr)
    print(header)
    short_actual = ["Bio", "Scr", "Des"]
    for i in range(n):
        label = short_actual[i] if i < len(short_actual) else class_names[i][:3]
        row = f"  Actual {label:<4}"
        for j in range(n):
            row += f"  {cm[i][j]:>10}"
        print(row)


def print_logistic_importance(model, feature_names, binary_mode, class_names):
    """Print the top logistic regression coefficients per class.

    LR coefficients are direct linear weights and are biologically interpretable: a large
    positive coefficient for feature X in the Designed class means sequences with high X
    values are strongly pushed toward a Designed prediction. For example, a high coefficient
    on the "WW" dipeptide for Designed would suggest Trp-Trp adjacency is a binder
    signature. Features with large coefficients across multiple classes simultaneously are
    compositionally ambiguous — the sign tells you which class is favored.
    """
    coef = model.coef_
    if binary_mode:
        coefs = coef[0]
        order_pos = np.argsort(coefs)[::-1]
        order_neg = np.argsort(coefs)
        print("  Top 10 features pushing toward Scrambled (positive coefficients):")
        for rank, idx in enumerate(order_pos[:10]):
            print(f"    {rank + 1:2}. {feature_names[idx]}: {coefs[idx]:+.6f}")
        print("  Top 10 features pushing toward Biological (most negative coefficients):")
        for rank, idx in enumerate(order_neg[:10]):
            print(f"    {rank + 1:2}. {feature_names[idx]}: {coefs[idx]:+.6f}")
    else:
        for k, cname in enumerate(class_names):
            coefs = coef[k]
            pos_idx = np.where(coefs > 0)[0]
            if len(pos_idx) == 0:
                print(f"  Class {cname}: no positive coefficients.")
                continue
            top_local = pos_idx[np.argsort(coefs[pos_idx])[::-1][:10]]
            print(f"  Class {cname} — top 10 features with highest positive coefficients:")
            for rank, idx in enumerate(top_local):
                print(f"    {rank + 1:2}. {feature_names[idx]}: {coefs[idx]:+.6f}")


def print_random_forest_importance(model, feature_names, top_n=20):
    """Print the top features by mean decrease in impurity (MDI) importance.

    Used for both RandomForest and XGBoost since both expose feature_importances_.
    MDI ranks features by how much they reduce class uncertainty across all splits in the
    ensemble. Features that rank highly across both tree models and LR are the most robust
    discriminators — they matter regardless of inductive bias. Dipeptide features are
    expected to dominate for bio-vs-scrambled; physicochemical features may dominate for
    bio-vs-designed since the AI design process can alter global sequence chemistry.
    """
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    print(f"  Top {top_n} features by importance:")
    for rank, idx in enumerate(order):
        print(f"    {rank + 1:2}. {feature_names[idx]}: {imp[idx]:.6f}")


def print_binder_prediction_table(seq_ids, y_pred, proba, class_names):
    """Print per-binder class probabilities, sorted by P(Biological) descending.

    Sorting by P(Biological) identifies the designed binders whose sequence statistics most
    closely resemble natural proteins — the "most natural-looking" designs. A high
    P(Biological) score may indicate a design that successfully recapitulates natural protein
    sequence grammar, which is relevant for assessing how well the AI design tool generalizes
    beyond its training distribution. Conversely, binders with high P(Designed) are the most
    anomalous relative to natural sequence space — interesting candidates for understanding
    what makes computational design mechanistically distinct.
    """
    order = np.argsort(-proba[:, 0])
    header = (
        f"  {'Sequence ID':<42} {'Predicted':<14} "
        f"{'P(Biological)':>14} {'P(Scrambled)':>14} {'P(Designed)':>14}"
    )
    print(header)
    for i in order:
        sid = seq_ids[i] if len(str(seq_ids[i])) <= 40 else str(seq_ids[i])[:37] + "..."
        pred_name = class_names[y_pred[i]]
        pb, ps, pd_ = proba[i, 0], proba[i, 1], proba[i, 2]
        print(
            f"  {sid:<42} {pred_name:<14} "
            f"{pb:>14.4f} {ps:>14.4f} {pd_:>14.4f}"
        )


def print_model_comparison_table(reports, accuracies, model_names, class_names):
    """Print a side-by-side comparison table of all trained models.

    F1 macro weights all classes equally regardless of size, which is appropriate here
    because the three classes are imbalanced (500 bio, 500 scrambled, 110 designed).
    Per-class F1(Designed) is the primary scientific metric: successfully identifying
    AI-designed binders is the core research question, and a model that achieves high
    overall accuracy by ignoring the small Designed class is not scientifically useful.
    """
    print(f"  {'Model':<22} {'Accuracy':>10}  {'Precision(macro)':>16}  {'Recall(macro)':>13}  {'F1(macro)':>9}")
    for name, report, acc in zip(model_names, reports, accuracies):
        prec = report['macro avg']['precision']
        rec  = report['macro avg']['recall']
        f1   = report['macro avg']['f1-score']
        print(f"  {name:<22} {acc:>9.2%}  {prec:>16.2f}  {rec:>13.2f}  {f1:>9.2f}")
    print()
    header = f"  {'Model':<22}"
    for cn in class_names:
        header += f"  {'F1(' + cn + ')':>14}"
    print(header)
    for name, report in zip(model_names, reports):
        row = f"  {name:<22}"
        for cn in class_names:
            row += f"  {report[cn]['f1-score']:>14.2f}"
        print(row)
    print()



if __name__ == "__main__":
    # Main
    parser = argparse.ArgumentParser(
        description="Classify biological vs scrambled (and optionally designed binder) proteins from FASTA."
    )
    parser.add_argument(
        "--fasta",
        "-f",
        default=DEFAULT_FASTA_PATH,
        help="Path to FASTA file (.fasta or .fasta.gz)",
    )
    parser.add_argument(
        "--binders",
        "-b",
        default=None,
        help="Optional path to non-gzipped FASTA of designed binder sequences (adds Class 2)",
    )
    args = parser.parse_args()
    fasta_path = args.fasta
    binders_path = args.binders

    three_class = binders_path is not None
    class_names = ["Biological", "Scrambled", "Designed"] if three_class else ["Biological", "Scrambled"]

    # Step 1: Load data
    print("Step 1: Load data")
    print(f"  FASTA file: {fasta_path}\n")

    bio_sequences = load_fasta(fasta_path, max_sequences=500, min_length=50)
    if not bio_sequences:
        raise SystemExit(f"No sequences loaded from {fasta_path}. Check path and format.")

    random.seed(42)
    scrambled_sequences = [scramble_sequence(seq) for seq in bio_sequences]
    n_bio = len(bio_sequences)

    binder_ids = []
    binder_sequences = []
    if three_class:
        binder_ids, binder_sequences = load_fasta_plain_with_ids(binders_path, min_length=50)
        if not binder_sequences:
            raise SystemExit(f"No sequences loaded from {binders_path}. Check path and format.")

    all_sequences = bio_sequences + scrambled_sequences + binder_sequences
    labels = [0] * n_bio + [1] * n_bio + ([2] * len(binder_sequences) if three_class else [])

    aa_freqs = compute_aa_freqs_from_sequences(bio_sequences)

    print(f"  Biological: {n_bio} sequences (real UniProt)")
    print(f"  Scrambled: {n_bio} sequences (shuffled biological sequences)")
    if three_class:
        print(f"  Designed: {len(binder_sequences)} sequences (binder FASTA)")
    print(f"  Mode: {'3-class (Biological / Scrambled / Designed)' if three_class else 'binary (Biological / Scrambled)'}")
    print("\n  Amino acid frequencies in biological data:")
    for aa in AMINO_ACIDS:
        print(f"    {aa}: {aa_freqs[aa]:.4f}")
    print()

    # Step 2: Feature extraction
    print("Step 2: Feature extraction")
    print("  Feature breakdown:")
    print("    Amino acid frequencies:  20")
    print("    Dipeptide frequencies:   400")
    print("    Physicochemical:         9")
    print("    Total:                   429")
    print()

    X = np.array([build_feature_vector(seq) for seq in all_sequences])
    y = np.array(labels)

    print(f"  Feature matrix: {X.shape[0]} sequences × {X.shape[1]} features\n")

    bio_avg = X[:n_bio, :20].mean(axis=0)
    scr_avg = X[n_bio : 2 * n_bio, :20].mean(axis=0)

    print("  Average amino acid frequencies by class (first 20 features):")
    if three_class:
        des_avg = X[2 * n_bio :, :20].mean(axis=0)
        print(f"  {'AA':<4} {'Biological':>12} {'Scrambled':>12} {'Designed':>12}")
        for i, aa in enumerate(AMINO_ACIDS):
            print(f"  {aa:<4} {bio_avg[i]:>12.4f} {scr_avg[i]:>12.4f} {des_avg[i]:>12.4f}")
    else:
        print(f"  {'AA':<4} {'Biological':>12} {'Scrambled':>12} {'Difference':>12}")
        for i, aa in enumerate(AMINO_ACIDS):
            diff = bio_avg[i] - scr_avg[i]
            print(f"  {aa:<4} {bio_avg[i]:>12.4f} {scr_avg[i]:>12.4f} {diff:>+12.4f}")
    print()

    # Step 3: Train and evaluate
    print("Step 3: Train and evaluate")
    print("  Splitting 80% train / 20% test (stratified); StandardScaler fit on train for logistic regression only.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    feature_names = get_feature_names()

    lr_model = LogisticRegression(
        max_iter=2000, random_state=42, class_weight="balanced"
    )
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb_model = XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric='mlogloss' if three_class else 'logloss',
        verbosity=0,
    )

    lr_model.fit(X_train_s, y_train)
    predictions_lr = lr_model.predict(X_test_s)
    accuracy_lr = (predictions_lr == y_test).mean()
    print("  --- LogisticRegression (scaled features) ---")
    print(f"  Accuracy: {accuracy_lr:.2%}\n")
    print("  Classification Report:")
    print(classification_report(y_test, predictions_lr, target_names=class_names))
    report_lr = classification_report(y_test, predictions_lr, target_names=class_names, output_dict=True)
    cm_lr = confusion_matrix(y_test, predictions_lr)
    print("  Confusion Matrix:")
    print_confusion_matrix_block(cm_lr, class_names)
    print()

    rf_model.fit(X_train, y_train)
    predictions_rf = rf_model.predict(X_test)
    accuracy_rf = (predictions_rf == y_test).mean()
    print("  --- RandomForestClassifier (unscaled features) ---")
    print(f"  Accuracy: {accuracy_rf:.2%}\n")
    print("  Classification Report:")
    print(classification_report(y_test, predictions_rf, target_names=class_names))
    report_rf = classification_report(y_test, predictions_rf, target_names=class_names, output_dict=True)
    cm_rf = confusion_matrix(y_test, predictions_rf)
    print("  Confusion Matrix:")
    print_confusion_matrix_block(cm_rf, class_names)
    print()

    xgb_model.fit(X_train, y_train)
    predictions_xgb = xgb_model.predict(X_test)
    accuracy_xgb = (predictions_xgb == y_test).mean()
    print("  --- XGBClassifier (unscaled features) ---")
    print(f"  Accuracy: {accuracy_xgb:.2%}\n")
    print("  Classification Report:")
    print(classification_report(y_test, predictions_xgb, target_names=class_names))
    report_xgb = classification_report(y_test, predictions_xgb, target_names=class_names, output_dict=True)
    cm_xgb = confusion_matrix(y_test, predictions_xgb)
    print("  Confusion Matrix:")
    print_confusion_matrix_block(cm_xgb, class_names)
    print()

    # Step 4: Feature importance
    print("Step 4: Feature importance")

    print("  LogisticRegression:")
    print_logistic_importance(lr_model, feature_names, binary_mode=not three_class, class_names=class_names)
    print()

    print("  RandomForestClassifier:")
    print_random_forest_importance(rf_model, feature_names, top_n=20)
    print()

    print("  XGBClassifier:")
    print_random_forest_importance(xgb_model, feature_names, top_n=20)
    print()

    # Step 5: Per-binder predictions (3-class only)
    if three_class:
        print("Step 5: Per-binder prediction table")
        X_binder = X[2 * n_bio :]
        X_binder_s = scaler.transform(X_binder)
        print("  LogisticRegression (scaled binder features):")
        proba_lr_b = lr_model.predict_proba(X_binder_s)
        pred_lr_b = lr_model.predict(X_binder_s)
        print_binder_prediction_table(binder_ids, pred_lr_b, proba_lr_b, class_names)
        print()
        print("  RandomForestClassifier (unscaled binder features):")
        proba_rf_b = rf_model.predict_proba(X_binder)
        pred_rf_b = rf_model.predict(X_binder)
        print_binder_prediction_table(binder_ids, pred_rf_b, proba_rf_b, class_names)
        print()
        print("  XGBClassifier (unscaled binder features):")
        proba_xgb_b = xgb_model.predict_proba(X_binder)
        pred_xgb_b  = xgb_model.predict(X_binder)
        print_binder_prediction_table(binder_ids, pred_xgb_b, proba_xgb_b, class_names)
        print()

    print("Model Comparison")
    print_model_comparison_table(
        [report_lr, report_rf, report_xgb],
        [accuracy_lr, accuracy_rf, accuracy_xgb],
        ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"],
        class_names,
    )
