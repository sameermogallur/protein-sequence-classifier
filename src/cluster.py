"""
Unsupervised clustering of biological, scrambled, and designed protein sequences.

Scientific goal: reveal natural groupings in the 429-feature sequence space and
show where the 110 designed TolA binders land relative to biological sub-clusters
(membrane / soluble / disordered proteins are expected to form distinct groups).
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from train import (
    build_feature_vector,
    load_fasta,
    load_fasta_plain_with_ids,
    scramble_sequence,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def build_feature_matrix(bio_sequences, scrambled_sequences, designed_sequences):
    """Build the 429-feature matrix for all sequences and return it with class labels.

    Class labels: 0=Biological, 1=Scrambled, 2=Designed.
    Order within each class is preserved so indices can be mapped back to IDs.
    """
    all_seqs = bio_sequences + scrambled_sequences + designed_sequences
    n_bio = len(bio_sequences)
    n_scr = len(scrambled_sequences)
    n_des = len(designed_sequences)

    X = np.array([build_feature_vector(seq) for seq in all_seqs])
    class_labels = np.array([0] * n_bio + [1] * n_scr + [2] * n_des)
    return X, class_labels


def find_elbow_k(inertias, k_range):
    """Return best k using maximum second derivative of the inertia curve.

    The second derivative peaks where the inertia curve bends most sharply —
    the classic elbow. No external dependencies required.
    """
    deltas = np.diff(inertias)
    accel = np.diff(deltas)
    # accel[0] corresponds to the change at k_range[2] (third k value)
    best_idx = int(np.argmax(accel)) + 2
    return k_range[best_idx]


def run_elbow(X_scaled, k_range):
    """Fit K-means for each k in k_range and return inertias."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias


def save_elbow_plot(k_range, inertias, best_k, out_path):
    """Save elbow plot with a vertical line marking best_k."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, inertias, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Selected k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("K-means elbow method — 429-feature protein space")
    ax.set_xticks(k_range)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Elbow plot saved to {out_path}")


def save_pca_plot(X_2d, class_labels, cluster_labels, best_k,
                  var1, var2, out_path):
    """Save PCA scatter plot with class shape and cluster color encoding.

    Shape encodes sequence origin (bio/scrambled/designed).
    Color encodes K-means cluster assignment, consistent across classes.
    Designed binders are plotted last so they render on top.
    """
    cmap = plt.colormaps["tab10"].resampled(best_k)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot bio and scrambled first, designed last (higher zorder)
    class_specs = [
        (0, "o",  20,  0.45, "none",    "Biological"),
        (1, "^",  20,  0.30, "none",    "Scrambled"),
        (2, "*",  220, 1.00, "black",   "Designed binders"),
    ]

    for cls_idx, marker, size, alpha, edgecolor, label in class_specs:
        mask = class_labels == cls_idx
        zorder = 5 if cls_idx == 2 else 2
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=cluster_labels[mask],
            cmap=cmap, vmin=0, vmax=best_k - 1,
            marker=marker, s=size, alpha=alpha,
            edgecolors=edgecolor, linewidths=0.4,
            zorder=zorder,
            label=label,
        )

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance explained)")
    ax.set_title(f"PCA of 429-feature protein space — k={best_k} clusters")

    # Legend: class shapes
    shape_handles = [
        plt.scatter([], [], marker="o",  color="grey",   s=30,  alpha=0.6,  label="Biological"),
        plt.scatter([], [], marker="^",  color="grey",   s=30,  alpha=0.4,  label="Scrambled"),
        plt.scatter([], [], marker="*",  color="grey",   s=120, alpha=1.0,
                    edgecolors="black", linewidths=0.4,  label="Designed binders"),
    ]
    # Legend: cluster colors
    cluster_handles = [
        mpatches.Patch(color=cmap(k / max(best_k - 1, 1)), label=f"Cluster {k}") for k in range(best_k)
    ]

    leg1 = ax.legend(handles=shape_handles, title="Sequence origin",
                     loc="upper left", fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=cluster_handles, title="K-means cluster",
              loc="upper right", fontsize=8, title_fontsize=8)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PCA plot saved to {out_path}")


def print_cluster_summary(cluster_labels, class_labels, designed_ids, best_k):
    """Print per-cluster breakdown: count of bio/scrambled/designed + binder IDs.

    This lets us identify which clusters attract designed binders and infer
    whether those binders look soluble, membrane-like, or disordered.
    """
    print("\nCluster composition summary:")
    header = f"  {'Cluster':>8}  {'Biological':>12}  {'Scrambled':>12}  {'Designed':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    des_mask = class_labels == 2

    for k in range(best_k):
        in_cluster = cluster_labels == k
        n_bio = int(((class_labels == 0) & in_cluster).sum())
        n_scr = int(((class_labels == 1) & in_cluster).sum())
        n_des = int((des_mask & in_cluster).sum())
        print(f"  {k:>8}  {n_bio:>12}  {n_scr:>12}  {n_des:>10}")

    print()
    print("  Designed binder cluster assignments:")
    des_indices = np.where(des_mask)[0]
    for i, idx in enumerate(des_indices):
        seq_id = designed_ids[i] if i < len(designed_ids) else f"binder_{i}"
        k = cluster_labels[idx]
        print(f"    {seq_id:<40}  cluster {k}")


def label_clusters_by_physicochemistry(X_bio, cluster_labels, best_k):
    """Compute mean physicochemical stats per bio cluster and assign interpretive labels.

    Uses raw (unscaled) feature values so means are biologically interpretable.
    Feature indices: instability_index=422, gravy=424, net_charge_per_residue=428.
    Labeling heuristic: GRAVY>0 → Membrane-like; instability>40 → Disordered; else → Soluble.
    """
    IDX_INSTABILITY = 422
    IDX_GRAVY = 424
    IDX_NET_CHARGE = 428

    print("\n  Cluster physicochemical profile:")
    print(f"  {'Cluster':>8}  {'N':>5}  {'GRAVY':>8}  {'Instability':>12}  {'Net charge/res':>15}  Label")
    print("  " + "-" * 68)

    cluster_names = {}
    for k in range(best_k):
        mask = cluster_labels == k
        X_k = X_bio[mask]
        mean_gravy = X_k[:, IDX_GRAVY].mean()
        mean_inst = X_k[:, IDX_INSTABILITY].mean()
        mean_charge = X_k[:, IDX_NET_CHARGE].mean()

        if mean_gravy > 0.0:
            label = "Membrane-like"
        elif mean_inst > 40.0:
            label = "Disordered"
        else:
            label = "Soluble"

        cluster_names[k] = label
        print(f"  {k:>8}  {mask.sum():>5}  {mean_gravy:>8.3f}  {mean_inst:>12.1f}  {mean_charge:>15.4f}  {label}")

    return cluster_names


def save_bio_only_elbow_plot(k_range, inertias, best_k, out_path):
    """Save elbow plot for bio-only clustering."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, inertias, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Selected k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("K-means elbow — biological sequences only (429 features)")
    ax.set_xticks(k_range)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Elbow plot saved to {out_path}")


def save_bio_only_pca_plot(X_bio_2d, bio_cluster_labels, X_des_2d,
                            best_k, cluster_names, var1, var2, out_path):
    """Save PCA plot with bio sequences colored by cluster and designed binders projected in.

    Fitting PCA on bio-only sequences preserves the biological coordinate system;
    designed binders are transformed into that space to reveal which subtype they resemble.
    """
    cmap = plt.colormaps["tab10"].resampled(best_k)

    fig, ax = plt.subplots(figsize=(10, 7))

    for k in range(best_k):
        mask = bio_cluster_labels == k
        color = cmap(k / max(best_k - 1, 1))
        ax.scatter(
            X_bio_2d[mask, 0], X_bio_2d[mask, 1],
            color=color, marker="o", s=25, alpha=0.6,
            edgecolors="none", zorder=2,
        )

    ax.scatter(
        X_des_2d[:, 0], X_des_2d[:, 1],
        color="crimson", marker="*", s=220, alpha=1.0,
        edgecolors="black", linewidths=0.4, zorder=5,
    )

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance explained)")
    ax.set_title(f"Designed binders projected into biological PCA space — k={best_k} clusters")

    bio_handles = [
        mpatches.Patch(color=cmap(k / max(best_k - 1, 1)),
                       label=f"Cluster {k} — {cluster_names[k]}")
        for k in range(best_k)
    ]
    des_handle = plt.scatter([], [], marker="*", color="crimson", s=120,
                             edgecolors="black", linewidths=0.4, label="Designed binders")
    bio_handles.append(des_handle)

    ax.legend(handles=bio_handles, title="Biological clusters", fontsize=8, title_fontsize=8,
              loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PCA plot saved to {out_path}")


def print_bio_projection_summary(X_des_2d, X_bio_2d, bio_cluster_labels,
                                  designed_ids, best_k, cluster_names):
    """Print how many designed binders project nearest to each biological cluster.

    Nearest cluster is determined by Euclidean distance from each binder's 2D PCA
    point to the centroid of each bio cluster in the same 2D space.
    """
    centroids = np.array([
        X_bio_2d[bio_cluster_labels == k].mean(axis=0) for k in range(best_k)
    ])

    nearest = []
    for des_pt in X_des_2d:
        dists = np.linalg.norm(centroids - des_pt, axis=1)
        nearest.append(int(np.argmin(dists)))

    print("\n  Designed binders — nearest biological cluster (by PCA centroid distance):")
    for k in range(best_k):
        count = nearest.count(k)
        print(f"    Cluster {k} ({cluster_names[k]}):  {count} binders")

    print()
    print("  Per-binder assignments:")
    for i, (seq_id, k) in enumerate(zip(designed_ids, nearest)):
        print(f"    {seq_id:<50}  → Cluster {k} ({cluster_names[k]})")


def run_bio_only_analysis(bio_sequences, designed_sequences, designed_ids):
    """Cluster biological sequences alone and project designed binders into that space.

    This answers the core scientific question: which biological protein subtype
    (membrane / soluble / disordered) do the designed TolA binders most resemble?
    Keeping scrambled sequences out of the clustering ensures that natural biological
    structure drives the grouping rather than the composition-preserving noise control.
    """
    print("Building feature matrices (bio-only + designed)...")
    X_bio = np.array([build_feature_vector(seq) for seq in bio_sequences])
    X_des = np.array([build_feature_vector(seq) for seq in designed_sequences])
    print(f"  Bio matrix:      {X_bio.shape[0]} × {X_bio.shape[1]}")
    print(f"  Designed matrix: {X_des.shape[0]} × {X_des.shape[1]}")

    scaler_bio = StandardScaler()
    X_bio_scaled = scaler_bio.fit_transform(X_bio)
    X_des_scaled = scaler_bio.transform(X_des)

    k_range = list(range(2, 9))
    print(f"\nElbow method (k = {k_range[0]} to {k_range[-1]}, bio sequences only)...")
    inertias = run_elbow(X_bio_scaled, k_range)
    best_k = find_elbow_k(inertias, k_range)
    print(f"  Best k: {best_k}")
    save_bio_only_elbow_plot(
        k_range, inertias, best_k,
        os.path.join(FIGURES_DIR, "clustering_bio_only_elbow.png"),
    )

    print(f"\nK-means clustering (k={best_k}, bio sequences only)...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    bio_cluster_labels = km.fit_predict(X_bio_scaled)
    print(f"  Cluster sizes: { {k: int((bio_cluster_labels == k).sum()) for k in range(best_k)} }")

    cluster_names = label_clusters_by_physicochemistry(X_bio, bio_cluster_labels, best_k)

    print("\nPCA (2 components, fit on bio-only)...")
    pca = PCA(n_components=2, random_state=42)
    X_bio_2d = pca.fit_transform(X_bio_scaled)
    X_des_2d = pca.transform(X_des_scaled)
    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var1:.1f}%  PC2: {var2:.1f}%  (total: {var1 + var2:.1f}%)")

    print("\nSaving bio-only PCA plot...")
    save_bio_only_pca_plot(
        X_bio_2d, bio_cluster_labels, X_des_2d, best_k,
        cluster_names, var1, var2,
        os.path.join(FIGURES_DIR, "clustering_bio_only_pca.png"),
    )

    print_bio_projection_summary(
        X_des_2d, X_bio_2d, bio_cluster_labels, designed_ids, best_k, cluster_names
    )


def run_physicochemical_only_analysis(bio_sequences, designed_sequences, designed_ids):
    """Cluster biological sequences on the 9 physicochemical features only (indices 420-428).

    Hypothesis: in the 429-feature space, dipeptide features (400) contribute 93.2% of
    K-means distance signal after StandardScaler, making GRAVY and instability index
    geometrically invisible. Restricting to the 9 physicochemical features gives them
    equal weight, which should allow membrane/soluble/disordered structure to emerge if
    it exists in this sample.
    """
    PHYSICO_SLICE = slice(420, 429)

    print("Building feature matrices (bio-only + designed, physicochemical slice)...")
    X_bio = np.array([build_feature_vector(seq) for seq in bio_sequences])
    X_des = np.array([build_feature_vector(seq) for seq in designed_sequences])

    X_bio_p = X_bio[:, PHYSICO_SLICE]   # (500, 9)
    X_des_p = X_des[:, PHYSICO_SLICE]   # (110, 9)
    print(f"  Bio matrix (physico):      {X_bio_p.shape[0]} × {X_bio_p.shape[1]}")
    print(f"  Designed matrix (physico): {X_des_p.shape[0]} × {X_des_p.shape[1]}")

    scaler_p = StandardScaler()
    X_bio_ps = scaler_p.fit_transform(X_bio_p)
    X_des_ps = scaler_p.transform(X_des_p)

    k_range = list(range(2, 9))
    print(f"\nElbow method (k = {k_range[0]} to {k_range[-1]}, 9 physicochemical features)...")
    inertias = run_elbow(X_bio_ps, k_range)
    best_k = find_elbow_k(inertias, k_range)
    print(f"  Best k: {best_k}")

    elbow_path = os.path.join(FIGURES_DIR, "clustering_physico_only_elbow.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, inertias, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5, label=f"Selected k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("K-means elbow — 9 physicochemical features only")
    ax.set_xticks(k_range)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)
    print(f"  Elbow plot saved to {elbow_path}")

    print(f"\nK-means clustering (k={best_k}, physicochemical features only)...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    bio_cluster_labels = km.fit_predict(X_bio_ps)
    print(f"  Cluster sizes: { {k: int((bio_cluster_labels == k).sum()) for k in range(best_k)} }")

    # Pass full X_bio so label_clusters_by_physicochemistry can use absolute indices 422/424/428
    cluster_names = label_clusters_by_physicochemistry(X_bio, bio_cluster_labels, best_k)

    print("\nPCA (2 components, fit on 9 physicochemical features, bio-only)...")
    pca = PCA(n_components=2, random_state=42)
    X_bio_2d = pca.fit_transform(X_bio_ps)
    X_des_2d = pca.transform(X_des_ps)
    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var1:.1f}%  PC2: {var2:.1f}%  (total: {var1 + var2:.1f}%)")

    print("\nSaving physicochemical-only PCA plot...")
    save_bio_only_pca_plot(
        X_bio_2d, bio_cluster_labels, X_des_2d, best_k,
        cluster_names, var1, var2,
        os.path.join(FIGURES_DIR, "clustering_physico_only_pca.png"),
    )

    print_bio_projection_summary(
        X_des_2d, X_bio_2d, bio_cluster_labels, designed_ids, best_k, cluster_names
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised clustering and PCA visualization of protein sequences."
    )
    parser.add_argument("--fasta", "-f", required=True,
                        help="Path to gzipped biological FASTA (Swiss-Prot reviewed)")
    parser.add_argument("--binders", "-b", required=True,
                        help="Path to designed binder FASTA")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Step 1: Load data (mirrors train.py exactly) ---
    print("Step 1: Loading sequences")
    bio_sequences = load_fasta(args.fasta, max_sequences=500, min_length=50)
    if not bio_sequences:
        raise SystemExit(f"No sequences loaded from {args.fasta}")

    random.seed(42)
    scrambled_sequences = [scramble_sequence(seq) for seq in bio_sequences]

    designed_ids, designed_sequences = load_fasta_plain_with_ids(args.binders, min_length=50)
    if not designed_sequences:
        raise SystemExit(f"No sequences loaded from {args.binders}")

    print(f"  Biological:  {len(bio_sequences)}")
    print(f"  Scrambled:   {len(scrambled_sequences)}")
    print(f"  Designed:    {len(designed_sequences)}")
    print(f"  Total:       {len(bio_sequences) + len(scrambled_sequences) + len(designed_sequences)}")

    # --- Step 2: Build 429-feature matrix ---
    print("\nStep 2: Building 429-feature matrix")
    X, class_labels = build_feature_matrix(bio_sequences, scrambled_sequences, designed_sequences)
    print(f"  Feature matrix: {X.shape[0]} × {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Step 3: Elbow method ---
    print("\nStep 3: Elbow method (k = 2 to 10)")
    k_range = list(range(2, 11))
    inertias = run_elbow(X_scaled, k_range)
    best_k = find_elbow_k(inertias, k_range)
    print(f"  Best k (max second derivative): {best_k}")

    elbow_path = os.path.join(FIGURES_DIR, "clustering_elbow.png")
    save_elbow_plot(k_range, inertias, best_k, elbow_path)

    # --- Step 4: K-means at best_k ---
    print(f"\nStep 4: K-means clustering (k={best_k})")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    print(f"  Cluster sizes: { {k: int((cluster_labels == k).sum()) for k in range(best_k)} }")

    # --- Step 5: PCA ---
    print("\nStep 5: PCA (2 components)")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var1:.1f}%  PC2: {var2:.1f}%  (total: {var1 + var2:.1f}%)")

    # --- Step 6: Save PCA plot ---
    print("\nStep 6: Saving PCA plot")
    pca_path = os.path.join(FIGURES_DIR, "clustering_pca.png")
    save_pca_plot(X_2d, class_labels, cluster_labels, best_k,
                  var1, var2, pca_path)

    # --- Step 7: Cluster summary ---
    print_cluster_summary(cluster_labels, class_labels, designed_ids, best_k)

    # --- Analysis 2: Bio-only clustering + designed binder projection ---
    print("\n" + "=" * 60)
    print("Bio-only clustering: natural subtypes in biological sequences")
    print("=" * 60 + "\n")
    run_bio_only_analysis(bio_sequences, designed_sequences, designed_ids)

    # --- Analysis 3: Physicochemical-only clustering ---
    print("\n" + "=" * 60)
    print("Physicochemical-only clustering (9 features, indices 420-428)")
    print("Hypothesis: dipeptide dominance masked membrane/soluble/disordered structure")
    print("=" * 60 + "\n")
    run_physicochemical_only_analysis(bio_sequences, designed_sequences, designed_ids)


if __name__ == "__main__":
    main()
