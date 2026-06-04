"""
Author:
Aaro Kuusinen
kuusina2

Special Assignment
Spring 2026
"""

import PARAM
import time
import fasttext
import stanza
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from datetime import datetime
import platform, sys


# ------------------ Utils ------------------

def is_alpha_token(w):
    return w.isalpha() and len(w) >= PARAM.MIN_WORD_LEN and len(w) <= PARAM.MAX_WORD_LEN


# ------------------ Resources ------------------

def load_resources():
    """
    Make sure that you've downloaded the trained FastText models to the path:
    corpora/fasttext/<model>
    """
    print("Getting FastText")
    ft = fasttext.load_model(str(Path("corpora/fasttext") / f"cc.{PARAM.LANG}.300.bin"))
    print("Got FastText\n")

    print("Getting Stanza")
    stanza.download(
        PARAM.LANG, 
        processors="tokenize,pos,lemma",
        verbose=False,
        model_dir="corpora/stanza"
    )
    nlp = stanza.Pipeline(
        lang=PARAM.LANG,
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=True,
        verbose=False,
        model_dir="corpora/stanza"
    )
    print("Got Stanza\n")
    return ft, nlp


# ------------------ Vocabulary ------------------

def lemmatize(ws, nlp, batch, keep_upos=None, target_count=PARAM.TOTAL_WORDS):
    """
    ws: list of words
    nlp: language processor (Stanza)
    batch: batch size for the lemmatizer
    keep_upos: which word type to keep
    target_count: how many words in total to aim for for the pipeline to look at

    Returns a lemmatized list of given words in ws
    """
    keep_upos = keep_upos or {PARAM.WORD_TYPE}
    lemmas, seen = [], set()
    for i in range(0, len(ws), batch):
        chunk = ws[i:i+batch]
        doc = nlp([chunk])
        for sent in doc.sentences:
            for tok in sent.words:
                if tok.upos in keep_upos:
                    lem = (tok.lemma or tok.text).lower()
                    if is_alpha_token(lem) and lem not in seen:
                        seen.add(lem)
                        lemmas.append(lem)
        if target_count is not None and len(lemmas) >= target_count:
            break
    if target_count is not None and len(lemmas) > target_count:
        lemmas = lemmas[:target_count]
    return lemmas


def prepare_vocabulary(ft, nlp):
    """
    ft: fasttext model
    nlp: language processor (Stanza)

    Returns throroughly filtered and processed list of words from ft as a list
    """
    words = [w for w in ft.get_words() if is_alpha_token(w)]
    print(f"The entire vocabulary has (quickly filtered) {len(words)} words.\n")
    print(f"Sample words after quick filtering:\n" + "\n".join(words[:15]))
    candidate_words = lemmatize(words, nlp, PARAM.LEMMA_BATCH_SIZE, keep_upos=None)
    if PARAM.TOTAL_WORDS:
        candidate_words = candidate_words[:PARAM.TOTAL_WORDS]
    print(f"\nAfter lemmatization {len(candidate_words)} words.")
    print(f"\nSample candidate words:\n" + "\n".join(candidate_words[:15]))
    return candidate_words


def remove_words(ws, X, targets, w2i):
    """
    ws: list of all words
    X: vectors of the words
    targets: words to remove
    w2i: word to index mapping

    Returns updated word list, array of their vectors and word to index mapping
    """
    to_remove = [w2i[t] for t in targets if t in w2i]
    if not to_remove:
        print("No words to be removed from the candidate pool were found.")
        return ws, X, w2i
    removed_idx = np.array(sorted(set(to_remove)), dtype=int)
    mask = np.ones(len(ws), dtype=bool)
    mask[removed_idx] = False
    ws_f = [w for w, keep in zip(ws, mask) if keep]
    X_f = X[mask]
    w2i_f = {w: i for i, w in enumerate(ws_f)}
    return ws_f, X_f, w2i_f


# ------------------ Embeddings and Spaces ------------------

def build_embeddings(ft, words):
    """
    ft: FastText model
    words: list of words

    Returns an array of normalized word vectors.
    """
    X = np.vstack([ft.get_word_vector(w) for w in words]).astype(np.float32)
    return normalize(X, norm="l2")


def reduce_cluster_space(X, n_components=75, random_state=42, use_pca=True):
    """
    X: word vectors
    n_components: how many PCA components to keep (default 75)
    use_pca: Whether to use PCA or not

    Returns PCA reduced vectors, the PCA itself and it's explanation power.
    """
    if not use_pca:
        return X, None
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    explained = np.sum(pca.explained_variance_ratio_)
    print(f"\nCumulative explained variance ({n_components} PCs): {explained:.2%}")
    return X_pca, pca, explained


def build_viz_space(X_cluster, random_state=42):
    """
    X_cluster: Normalized pca space word vectors

    Returns the reduced t-SNE space
    """
    t0 = time.time()
    n = len(X_cluster)
    perp = max(5, min(40, n - 1))
    tsne = TSNE(n_components=2, verbose=1, perplexity=perp, max_iter=1000, random_state=random_state)
    tsne_results = tsne.fit_transform(X_cluster)
    print(f"t-SNE done in {time.time()-t0:.1f}s")
    return tsne_results


# ------------------ KMeans + Metrics ------------

def k_means_elbow_graph(X_cluster):
    """
    X_cluster: Normalized pca space word vectors
    """
    n = len(X_cluster)
    ks = [k for k in range(10, min(1010, n), 50)]
    wcss = []
    for i in ks:
        kmeans = KMeans(i, n_init=10, random_state=42)
        kmeans.fit(X_cluster)
        wcss.append(kmeans.inertia_)
    plt.figure()
    plt.plot(ks, wcss)
    plt.title(f'Elbow graph {PARAM.LANG}')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.tight_layout()
    plt.savefig(f"{PARAM.PLOT_PATH}/elbow_{PARAM.LANG}.png", dpi=200)
    plt.close()


def silhouette_graph(X_cluster):
    """
    X_cluster: Normalized pca space word vectors
    """
    n = len(X_cluster)
    ks, coeffs = [], []
    for k in range(10, min(1010, n), 50):
        if k < 2: break
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42).fit(X_cluster)
        ks.append(k)
        coeffs.append(silhouette_score(X_cluster, kmeans.labels_))
    plt.figure()
    plt.plot(ks, coeffs)
    plt.title(f"Silhouette (cluster space) {PARAM.LANG}")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.tight_layout()
    plt.savefig(f"{PARAM.PLOT_PATH}/silhouette_{PARAM.LANG}.png", dpi=200)
    plt.close()


def run_kmeans(X_cluster, k, init_centroids=None, random_state=42):
    """
    X_cluster: Word vectors
    k: number of clusters
    init_centroids: initial suggested centroid locations if there is

    Returns the labels and centroids of each cluster
    """
    if init_centroids is not None:
        km = KMeans(n_clusters=k, init=init_centroids, n_init=1, max_iter=50, random_state=random_state)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_cluster)
    # centers = v_norm(km.cluster_centers_)
    centers = km.cluster_centers_
    return labels, centers


def compute_ranks(ws):
    """
    ws: words

    Returns a list of ranks of the words. Aka in which order they appeared.
    """
    return np.array(np.arange(len(ws)), dtype=np.int64)


def cluster_order_by_freq(labels, ranks, stat='median'):
    """
    labels: labels of the clusters
    ranks: word ranks of the entire vocabulary
    stat: by which statistic to rank the clusters based on the word ranks

    Returns an ordered list of the cluster labels
    """
    order = []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        score = np.median(ranks[idxs]) if stat == "median" else np.min(ranks[idxs])
        order.append((float(score), int(c)))
    order.sort(key=lambda t: t[0])
    return [c for _, c in order]


# ----------------- Word Ops ------------------

def top_words_for_centroid(center, words, X, topn):
    """
    center: cluster centroid
    words: list of words in the vocabulary
    X: vectors of the words
    topn: how many words to show from the cluster

    Returns the relative ranking of the words in relation to the centroid, and the words in that cluster
    """
    # sims = Xnorm @ center
    sims = cosine_distances(X, center.reshape(1, -1)).ravel()
    k = min(topn, len(sims))
    if k == 0:
        return np.array([], dtype=int), []
    idxs = np.argpartition(sims, k-1)[:k]
    idxs = idxs[np.argsort(sims[idxs])]
    idxs_rel = np.arange(len(idxs))
    return idxs_rel, [words[i] for i in idxs]


def nearest_word(X, v):
    """
    X: word vectors of all words
    v: vector of the seed word

    Returns the index of the most similar word and its similarity
    """
    # sims = Xnorm @ v_normed
    sims = cosine_distances(X, v.reshape(1, -1)).ravel()
    i = int(np.argmax(sims))
    return i, float(sims[i])


# ------------------ Plotting ------------------

def build_plot_df(ws, tsne, labels=None):
    """
    ws: List of words
    tsne: t-SNE space

    Returns a ready to plot dataframe from the tsne space
    """
    df = pd.DataFrame({
        "word": ws,
        "tsne1": tsne[:, 0],
        "tsne2": tsne[:, 1],
        "label": labels if labels is not None else [""]*len(ws)
    })
    return df


def plot_points(df, out_path, title, alpha=0.35, show_legend=False):
    """
    df: plottable data in a DataFrame
    out_path: path where to save the plot
    title: title of the plot
    alpha: opaqueness of the dots in the plot
    """
    plt.figure(figsize=(10, 8))
    if "label" in df.columns and df["label"].nunique() > 1:
        ax = sns.scatterplot(data=df, x="tsne1", y="tsne2", hue="label", palette="tab20", s=12, alpha=alpha, linewidth=0)
        if show_legend:
            ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
            ax.get_legend().remove()
    else:
        ax = sns.scatterplot(data=df, x="tsne1", y="tsne2", s=12, alpha=alpha, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_chosen_overlay(df, targets, out_path, annotate=True, color="crimson", label="targets"):
    """
    df: Dataframe of words
    targets: words to show
    out_path: where to save the plot
    annotate: which words to show
    """
    word2idx = {w: i for i, w in enumerate(df["word"])}
    idxs = [word2idx[t] for t in targets if t in word2idx]
    missing = [t for t in targets if t not in word2idx]
    if missing:
        print("Warning: missing in viz:", ", ".join(missing))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(df["tsne1"], df["tsne2"], s=8, color="#cccccc", alpha=0.4, linewidth=0)
    if idxs:
        xs = df["tsne1"].to_numpy()[idxs]
        ys = df["tsne2"].to_numpy()[idxs]
        plt.scatter(xs, ys, s=40, color=color, edgecolor="black", linewidth=0.6, alpha=0.9, marker="o", label=label)
        if annotate:
            for i, (x, y) in zip(idxs, zip(xs, ys)):
                plt.annotate(df["word"].iat[i], xy=(x, y), xytext=(5, 2), textcoords="offset points", ha='right', va='bottom', fontsize=8)
        plt.legend(loc="best")
    plt.title(f"t-SNE Final {label} {PARAM.LANG}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ------------------ Interaction ------------------

def pick_words(ws, targets):
    """
    ws: words to choose from
    targets: already chosen words

    Returns an appended targets list of picked words.
    """
    if PARAM.AUTOMATIC:
        take = min(PARAM.AUTO_WORDS_PER_CLUSTER, len(ws))
        targets.extend(ws[:take])
        return targets
    choices = input("Words to save (space-separated indices, first=0):\n").strip()
    if not choices:
        return targets
    for idx_s in choices.split():
        try:
            idx = int(idx_s)
            if 0 <= idx < len(ws):
                targets.append(ws[idx])
            else:
                print(f"Index out of range: {idx}")
        except ValueError:
            print(f"Not an integer: {idx_s}")
    return targets


def pick_rel_indices(n, prompt):
    """
    n: how many there are to choose from
    prompt: prompt to show to the user

    Returns list of chosen indices.
    """
    raw = input(prompt).strip()
    if not raw:
        return []
    out = []
    for tok in raw.split():
        try:
            i = int(tok)
            if 0 <= i < n:
                out.append(i)
            else:
                print(f"Relative index out of range: {i}")
        except ValueError:
            print(f"Not an integer: {tok}")
    return out


def choices_rel(ordered_cluster_ids, centroids, candidate_words, X, topn):
    """
    ordered_cluster_ids: ordered list of cluster ids based on word freqs.
    centroids: cluster centroids
    candidate_words: all words
    X: word vectors
    topn: how many clusters to present

    Returns list of picked target words.
    """
    print(f"\nAvailable clusters: {len(ordered_cluster_ids)}")
    if PARAM.AUTOMATIC:
        rels = list(range(min(PARAM.AUTO_CLUSTER_COUNT, len(ordered_cluster_ids))))
    else:
        rels = pick_rel_indices(
            len(ordered_cluster_ids),
            "\nClusters to save (space-separated first=0):\n"
        )
    targets = []
    for r in rels:
        c_abs = ordered_cluster_ids[r]
        idxs, top_ws = top_words_for_centroid(centroids[c_abs], candidate_words, X, topn)
        print(f"\nCluster {r}:\n" +
              "\n".join(f"{i}\t{w}" for (i, w) in zip(idxs, top_ws)) +
              "\n")
        if PARAM.AUTOMATIC:
            take = min(PARAM.AUTO_WORDS_PER_CLUSTER, len(top_ws))
            targets.extend(top_ws[:take])
        else:
            targets = pick_words(top_ws, targets)
    seen = set()
    targets = [w for w in targets if not (w in seen or seen.add(w))]
    return targets


def review_collect_words(rows, targets, second_primers):
    """
    rows: rows of triplets
    targets: target words
    second_primers: second primer words

    Returns an ordered list of all the words
    """
    p2s = [p2 for lst in second_primers.values() for p2 in lst]
    p1s = [r["primer1"] for r in rows]
    ws = set(targets) | set(p2s) | set(p1s)
    return sorted(ws)


def review_choose_words_to_delete(ws, batch_size=10):
    """
    ws: all chosen words
    batch_size: the viewable word batch size

    Returns words to delete
    """
    to_delete = set()
    total = len(ws)
    if total == 0:
        return to_delete
    
    print(f"\nReview picked words (total ({total}).\nEnter indices per batch to delete.\nEnter to skip batch, 'q' to finish.)")
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = ws[start:end]
        print(f"\nBatch {start // batch_size + 1} [{start}:{end}]")
        for i, w in enumerate(batch):
            print(f"{i:2d}  {w}")
        raw = input("Delete indices in this batch (space separated; Enter=skip; q=finish): ").strip()
        if not raw:
            continue
        if raw.lower() in {"q", "quit"}:
            break
        idxs = []
        for tok in raw.split():
            try:
                i = int(tok)
                if 0 <= i < len(batch):
                    idxs.append(i)
                else:
                    print(f"Index out of range for this batch: {i}")
            except ValueError:
                print(f"Not an integer: {tok}")
        for i in idxs:
            to_delete.add(batch[i])
    if to_delete:
        print("\nMarked for deletion:", ", ".join(sorted(to_delete)))
    else:
        print("\nNo words marked for deletion.")
    return to_delete


# ----------------- Distances ------------------

def compute_similarity_matrix(X, T):
    """
    X: all word vectors
    T: target word vectors

    Returns a distance matrix
    """
    return cosine_distances(X, T)


def sanitize_relative_bands(bands):
    """
    bands: intervals of the relative distance categories

    Returns surely formatted distance bands.
    """
    clean = []
    for lo, hi in bands:
        p_lo = 0.0 if lo is None else float(lo)
        p_hi = 1.0 if hi is None else float(hi)
        p_lo = max(0.0, min(1.0, p_lo))
        p_hi = max(0.0, min(1.0, p_hi))
        if p_lo > p_hi:
            p_lo, p_hi = p_hi, p_lo
        clean.append((p_lo, p_hi))
    # sort by lower bound
    clean.sort(key=lambda x: x[0])
    return clean


def rank_by_relative_bands(D_col, bands):
    """
    D_col: distance columns
    bands: distance bands

    Returns word indeces per distance band and where the cut off indeces.
    """
    N = len(D_col)
    order = np.argsort(D_col)  # closest first
    per_band_idxs, cuts = [], []

    for p_lo, p_hi in bands:
        # convert proportions to index cuts in [0, N]
        k_lo = int(np.floor(p_lo * N))
        k_hi = int(np.floor(p_hi * N))
        # clamp and ensure non-decreasing
        k_lo = max(0, min(N, k_lo))
        k_hi = max(k_lo, min(N, k_hi))
        # slice the sorted order; result remains sorted by distance
        idxs = order[k_lo:k_hi]
        per_band_idxs.append(idxs)
        cuts.append((k_lo, k_hi))
    return per_band_idxs, cuts


def build_band_df(ws, D_col, idxs_abs, band_id):
    """
    ws: all words
    D_col: Distance column
    idxs_abs: Absolute indices of the words
    band_id: distance band id

    Returns a ready DataFrame containing information about the word in relation to a target word
    """
    if len(idxs_abs) == 0:
        return pd.DataFrame(columns=["rel_idx", "idx_abs", "word", "dist", "band"])
    df = pd.DataFrame({
        "idx_abs": idxs_abs,
        "word": np.array(ws, dtype=object)[idxs_abs],
        "dist": D_col[idxs_abs],
        "band": band_id
    })
    df = df.reset_index(drop=True).rename_axis("rel_idx").reset_index()
    return df


def prepare_target_relative_band_tables(ws, X, targets, T, bands):
    """
    ws: list of all words
    X: word vectors
    targets: list of target words or other words to which to measure distances
    T: target word vectors
    bands: intervals of the relative distance categories

    Returns a list of band dataframes per target
    """
    D = compute_similarity_matrix(X, T)  # (N, M)
    bands_sane = sanitize_relative_bands(bands)
    per_target = {}
    for j, t in enumerate(targets):
        band_idxs, _ = rank_by_relative_bands(D[:, j], bands_sane)
        band_dfs = [build_band_df(ws, D[:, j], idxs, b_id)
                    for b_id, idxs in enumerate(band_idxs)]
        per_target[t] = band_dfs
    return per_target


def show_band_sample(df_band, per_band_limit=20):
    """
    df_band: Dataframe of the distance band
    per_band_limit: how many words to show from the band

    Returns the showcased words
    """
    if df_band.empty:
        print("  [No items in this band]")
        return df_band.iloc[0:0].copy()
    block = df_band.sort_values('idx_abs', ascending=True).head(per_band_limit).copy()
    block = block.reset_index(drop=True)
    block['disp_rel_idx'] = np.arange(len(block), dtype=int)
    b = int(block["band"].iat[0])
    print(f"\nBand {b}: showing top {min(per_band_limit, len(block))} items")
    for _, row in block.iterrows():
        print(f"{int(row['disp_rel_idx']):5d}  {row['word']:<20s}  dist={row['dist']:.6f}")
    return block


def pick_from_band(df_band, p1=False):
    """
    df_band: distance band dataframe of shown words
    p1: whether the case is about choosing the first primers or not

    Returns a list of selected words
    """
    if df_band.empty:
        return []
    if PARAM.AUTOMATIC:
        take = min((PARAM.AUTO_WORDS_PER_BAND if not p1 else 1), len(df_band))
        return df_band.head(take)["word"].tolist()
    idx_col = "disp_rel_idx" if "disp_rel_idx" in df_band.columns else "rel_idx"
    raw = input("Pick words by rel_idx in this band (space-separated, Enter to skip):\n").strip()
    if not raw:
        return []
    rel_idxs = []
    for tok in raw.split():
        try:
            i = int(tok)
            if 0 <= i < len(df_band):
                rel_idxs.append(i)
            else:
                print(f"Relative index out of range: {i}")
        except ValueError:
            print(f"Not an integer: {tok}")
    sel = df_band[df_band[idx_col].isin(rel_idxs)]
    return sel["word"].tolist()


# ------------------ Main Pipeline ------------------

def main():
    ft, nlp = load_resources()
    
    print("Starting pipeline\n")
    candidate_words = prepare_vocabulary(ft, nlp)
    X = build_embeddings(ft, candidate_words)       # Normalized data space

    # Choose cluster space: PCA or raw X
    X_cluster, pca, explained = reduce_cluster_space(X, n_components=PARAM.PCA_COUNT, use_pca=PARAM.PCA)

    if PARAM.CLUSTERING_GRAPHS:
        # Metrics computed on cluster space
        print("\nMaking silhouette plot")
        silhouette_graph(X_cluster)
        print("\nMaking elbow plot")
        k_means_elbow_graph(X_cluster)

    # Visualization space (t-SNE) from cluster space
    tsne_results = build_viz_space(X_cluster, random_state=42)
    df_plot = build_plot_df(candidate_words, tsne_results)

    # Basic t-SNE scatter
    plot_points(df_plot, f"{PARAM.PLOT_PATH}/tsne_{PARAM.LANG}.png", f"t-SNE {PARAM.LANG}", show_legend=False)
    
    word2idx = {w: i for i, w in enumerate(candidate_words)}


    # ---------- THE UNSEEDED CLUSTERS PATH ----------

    if not PARAM.SEEDS and not PARAM.TARGET_WORDS:
        if PARAM.AUTOMATIC and PARAM.CLUSTER_COUNT == 0:
            k = PARAM.CLUSTER_COUNT
        else:
            user_input = input(f"How many clusters? Press Enter for default ({PARAM.CLUSTER_COUNT}):\n").strip()
            k = int(user_input) if user_input else PARAM.CLUSTER_COUNT

        print("\nKMeans without seeds\n")
        labels, centers = run_kmeans(X_cluster, k)
        ranks = compute_ranks(candidate_words)

        freq_order = cluster_order_by_freq(labels, ranks, stat="median")
        
        print("\nPreview clusters:")
        # Show a few clusters
        for r, c_abs in enumerate(freq_order[:min(PARAM.VIEW_CLUSTERS, k)]):
            _, top_ws = top_words_for_centroid(centers[c_abs], candidate_words, X_cluster, PARAM.TOP_WORDS)
            print(f"\nCluster {r}:\n{'\n'.join(top_ws)}")
        
        # Color the t-SNE by cluster labels
        df_plot_lab = build_plot_df(candidate_words, tsne_results, labels=labels)
        plot_points(df_plot_lab, f"{PARAM.PLOT_PATH}/tsne_clusters_{PARAM.LANG}.png", f"t-SNE with unseeded KMeans {PARAM.LANG}", show_legend=False)

        # Interactive slection of targets
        targets = choices_rel(freq_order, centers, candidate_words, X_cluster, PARAM.TOP_WORDS)


    # ---------- THE SEEDED CLUSTERS PATH ----------

    elif PARAM.SEEDS and not PARAM.TARGET_WORDS:
        if PARAM.AUTOMATIC and PARAM.CLUSTER_COUNT == 0:
            k = PARAM.CLUSTER_COUNT
        else:
            user_input = input(f"How many clusters? Press Enter for default ({PARAM.CLUSTER_COUNT}):\n").strip()
            k = int(user_input) if user_input else PARAM.CLUSTER_COUNT

        print(f"\nKMeans with seeds\n")
        seed_centers, seed_info = [], []
        for s in PARAM.SEEDS:
            v = ft.get_word_vector(s).astype(np.float32)
            i_near, sim_near = nearest_word(X, v)
            near_word = candidate_words[i_near]
            center_v = X_cluster[i_near] if PARAM.SNAP_SEEDS else (pca.transform(v[None,:]) if pca is not None else v)
            seed_centers.append(center_v.squeeze())
            seed_info.append(("seed", s, near_word, sim_near))

        need_rand = k - len(seed_centers)
        rand_centers, rand_info = [], []
        if need_rand > 0:
            used_idx = set(word2idx[w] for _, _, w, _ in seed_info) if PARAM.SNAP_SEEDS else set()
            pool = np.array([i for i in range(len(candidate_words)) if i not in used_idx])
            chosen = pool[:need_rand]
            for i in chosen:
                rand_centers.append(X_cluster[i])
                rand_info.append(("rand", candidate_words[i], candidate_words[i], 1.0))

        init_centr = np.vstack(seed_centers + rand_centers).astype(np.float32)
        print("\nInitial centroids (type, source, anchor/nearest, cosine):")
        for t, src, anchor, sim in seed_info + rand_info:
            print(f" {t:4s} src='{src}' center='{anchor}' cos={sim:.3f}")

        labels, centers = run_kmeans(X_cluster, k, init_centroids=init_centr)
        
        ranks = compute_ranks(candidate_words)
        freq_order = cluster_order_by_freq(labels, ranks, stat="median")
        
        print("\nPreview clusters:")
        for r in range(min(PARAM.VIEW_CLUSTERS, k)):
            _, top_ws = top_words_for_centroid(centers[r], candidate_words, X_cluster, PARAM.TOP_WORDS)
            print(f"\nCluster {r}:\n{'\n'.join(top_ws)}")

        df_plot_lab = build_plot_df(candidate_words, tsne_results, labels=labels)
        plot_points(df_plot_lab, f"{PARAM.PLOT_PATH}/tsne_seeded_clusters_{PARAM.LANG}.png", f"t-SNE with seeded KMeans {PARAM.LANG}", show_legend=False)

        targets = choices_rel(freq_order, centers, candidate_words, X_cluster, PARAM.TOP_WORDS)


    # ---------- THE READY TARGETS PATH ----------
    
    else:
        targets = PARAM.TARGET_WORDS
        print("\nUsing provided target words:\n" + "\n".join(targets) + "\n")

    # Targets and distances
    if not targets:
        print("No targets were selected.")

    
    # ---------- SELECTION OF PRIMERS ----------

    else:
        target_vectors = np.array([ft.get_word_vector(w) for w in targets]).astype(np.float32)
        print(f"\nThe final target words are:\n{'\n'.join(targets)}\n")

        candidate_words, X, word2idx = remove_words(candidate_words, X, targets, word2idx)

        second_primers = {t: [] for t in targets}
        t_vecs = {t: v for t, v in zip(targets, target_vectors)}
        # used_p2 = set()
        p2_to_band = {}
        p2_to_target = {}

        for t in targets:
            print(f"\n=== Target: {t} ===")
            band_pos = 0
            while True:
                band_tables = prepare_target_relative_band_tables(candidate_words, X, [t], t_vecs[t][None, :], PARAM.DISTANCES)
                band_dfs = band_tables.get(t, [])
                if band_pos >= len(band_dfs):
                    break
                df_band = band_dfs[band_pos]
                if df_band.empty:
                    band_pos += 1
                    continue

                block = show_band_sample(df_band, per_band_limit=20)
                picks = pick_from_band(block)
                if picks:
                    print("Picked:", ", ".join(picks))
                    second_primers[t].extend(picks)
                    for w in picks:
                        p2_to_band[(t, w)] = band_pos
                        p2_to_target.setdefault(w, []).append(t)
                    # used_p2.update(new_picks)
                    # candidate_words, X, word2idx = remove_words(candidate_words, X, picks, word2idx)
                band_pos += 1

        all_second = [w for lst in second_primers.values() for w in lst]
        candidate_words, X, word2idx = remove_words(candidate_words, X, all_second, word2idx)

        rows = []
        if all_second:
            print(f"\nPicked words as 2nd primes:\n{'\n'.join(all_second)}")
            
            t_col = {t: j for j, t in enumerate(targets)}
            t_vecs = {t: target_vectors[t_col[t]] for t in targets}

            for p2 in all_second:
                band_pos = 0
                while True:
                    p2_vec = ft.get_word_vector(p2).astype(np.float32)[None, :]
                    band_tables = prepare_target_relative_band_tables(candidate_words, X, [p2], p2_vec, [PARAM.DISTANCES[0]])
                    band_dfs = band_tables.get(p2, [])
                    if band_pos >= len(band_dfs):
                        break

                    df_band = band_dfs[band_pos]
                    print(f"\n=== 2nd Primer: {p2} ===")
                    block = show_band_sample(df_band, per_band_limit=10)
                    p1_picks = pick_from_band(block, p1=True)
                    
                    if p1_picks:
                        p2_v = p2_vec[0]
                        t_a = p2_to_target.get(p2, [])
                        for t in t_a:
                            t_v = t_vecs[t]
                            p2_to_t = cosine_distances(t_v.reshape(1, -1), p2_v.reshape(1, -1))[0, 0]
                            dist_group_for_row = int(p2_to_band.get((t, p2), 0))

                            for p1 in p1_picks:
                                p1_v = ft.get_word_vector(p1).astype(np.float32)
                                d_p1_p2 = cosine_distances(p1_v.reshape(1, -1), p2_v.reshape(1, -1))[0, 0]
                                d_p1_t = cosine_distances(t_v.reshape(1, -1), p1_v.reshape(1, -1))[0, 0]

                                rows.append({
                                    "target": t,
                                    "primer2": p2,
                                    "primer1": p1,
                                    "dist_group": dist_group_for_row,
                                    "primer1_to_prime2": d_p1_p2,
                                    "primer1_to_target": d_p1_t,
                                    "primer2_to_target": p2_to_t,
                                    })
                            # candidate_words, X, word2idx = remove_words(candidate_words, X, p1_picks, word2idx)
                    band_pos += 1

            if rows:
                rows_before = len(rows)
                if PARAM.AUTOMATIC:
                    all_words = review_collect_words(rows, targets, second_primers)
                    words_to_delete = review_choose_words_to_delete(all_words, batch_size=10)
                    
                    df_out = pd.DataFrame(rows, columns=[
                        "target", "primer2", "primer1", "dist_group",
                        "primer1_to_prime2", "primer1_to_target", "primer2_to_target"
                    ])
                    df_out = df_out.drop_duplicates(subset=["target", "primer2", "primer1"]).reset_index(drop=True)
                    rows_before = len(df_out)
                    
                    if words_to_delete:
                        mask_bad = df_out[["target", "primer2", "primer1"]].isin(words_to_delete).any(axis=1)
                        removed_rows = int(mask_bad.sum())
                        df_out = df_out.loc[~mask_bad].reset_index(drop=True)
                        print(f"Removed {removed_rows} rows due to deleted words.")
                    
                    plot_chosen_overlay(df_plot, df_out["target"].tolist(), f"{PARAM.PLOT_PATH}/tsne_targets_{PARAM.LANG}.png", annotate=True)
                    plot_chosen_overlay(df_plot, df_out["primer2"].tolist(), f"{PARAM.PLOT_PATH}/tsne_2primers_{PARAM.LANG}.png", annotate=True, label="2nd primers")
                    plot_chosen_overlay(df_plot, df_out["primer1"].tolist(), f"{PARAM.PLOT_PATH}/tsne_1primers_{PARAM.LANG}.png", annotate=True, label="1st primers")

                    out_path = f"triplets_{PARAM.LANG}.csv"
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    df_out.to_csv(out_path, index=False)
                    print(f"\nSaved {len(df_out)} rows to: {out_path}")
                
                else:
                    out_path = f"triplets_{PARAM.LANG}.csv"
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    df_out = pd.DataFrame(rows, columns=[
                        "target", "primer2", "primer1", "dist_group",
                        "primer1_to_primer2", "primer1_to_target", "primer2_to_target"
                    ])
                    plot_chosen_overlay(df_plot, df_out["target"].tolist(), f"{PARAM.PLOT_PATH}/tsne_targets_{PARAM.LANG}.png", annotate=True)
                    plot_chosen_overlay(df_plot, df_out["primer2"].tolist(), f"{PARAM.PLOT_PATH}/tsne_2primers_{PARAM.LANG}.png", annotate=True, label="2nd primers")
                    plot_chosen_overlay(df_plot, df_out["primer1"].tolist(), f"{PARAM.PLOT_PATH}/tsne_1primers_{PARAM.LANG}.png", annotate=True, label="1st primers")
                    df_out.to_csv(out_path, index=False)
                    print(f"\nSaved {len(df_out)} rows to: {out_path}")
            else:
                print("\nNo triplets to save.")

            try:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = Path(PARAM.PLOT_PATH) / "runs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"run_{run_id}.txt"

                # Basic counts
                n_rows_after = len(df_out)
                n_targets = df_out["target"].nunique() if "target" in df_out.columns else None
                n_p2 = df_out["primer2"].nunique() if "primer2" in df_out.columns else None
                n_p1 = df_out["primer1"].nunique() if "primer1" in df_out.columns else None
                dist_counts = df_out["dist_group"].value_counts().to_dict() if "dist_group" in df_out.columns else {}

                # Try to capture k and labels if available
                k_used = locals().get("k", getattr(PARAM, "CLUSTER_COUNT", None))
                labels_used = locals().get("labels", None)
                clusters_found = int(np.unique(labels_used).size) if labels_used is not None else None

                # Candidate size if available
                cand_size = len(candidate_words) if "candidate_words" in locals() and candidate_words is not None else None

                # Library versions (optional)
                vers = {
                    "python": sys.version.replace("\n", " "),
                    "platform": platform.platform(),
                    "numpy": getattr(np, "__version__", "?"),
                    "pandas": getattr(pd, "__version__", "?"),
                    "sklearn": getattr(__import__("sklearn"), "__version__", "?"),
                    "stanza": getattr(stanza, "__version__", "?"),
                    "fasttext": getattr(fasttext, "__version__", "?"),
                    "matplotlib": getattr(plt, "__version__", "?"),
                    "seaborn": getattr(sns, "__version__", "?"),
                }

                lines = []
                lines.append(f"Run ID: {run_id}")
                lines.append(f"CSV: {out_path if 'out_path' in locals() else '(unknown)'}")
                lines.append("")
                lines.append("[General]")
                lines.append(f"  Language: {getattr(PARAM, 'LANG', '')}")
                lines.append(f"  Plot path: {getattr(PARAM, 'PLOT_PATH', '')}")
                lines.append("")
                lines.append("[Automation]")
                lines.append(f"  AUTOMATIC: {getattr(PARAM, 'AUTOMATIC', False)}")
                lines.append(f"  AUTO_CLUSTER_COUNT: {getattr(PARAM, 'AUTO_CLUSTER_COUNT', None)}")
                lines.append(f"  AUTO_WORDS_PER_CLUSTER: {getattr(PARAM, 'AUTO_WORDS_PER_CLUSTER', None)}")
                lines.append(f"  AUTO_WORDS_PER_BAND: {getattr(PARAM, 'AUTO_WORDS_PER_BAND', None)}")
                lines.append("")
                lines.append("[Clustering]")
                lines.append(f"  SEEDS: {', '.join(getattr(PARAM, 'SEEDS', [])) if getattr(PARAM, 'SEEDS', []) else '[]'}")
                lines.append(f"  SNAP_SEEDS: {getattr(PARAM, 'SNAP_SEEDS', False)}")
                lines.append(f"  CLUSTER_COUNT: {getattr(PARAM, 'CLUSTER_COUNT', None)}")
                lines.append(f"  k_used: {k_used}")
                lines.append(f"  PCA: {getattr(PARAM, 'PCA', False)}")
                lines.append(f"  PCA_COUNT: {getattr(PARAM, 'PCA_COUNT', None)}")
                lines.append(f"  PCA explains: {explained}")
                lines.append(f"  clusters_found: {clusters_found}")
                lines.append("")
                lines.append("[Bands]")
                lines.append(f"  DISTANCES: {getattr(PARAM, 'DISTANCES', [])}")
                lines.append("")
                lines.append("[Vocabulary]")
                lines.append(f"  candidate_words: {cand_size}")
                lines.append(f"  MIN_WORD_LEN: {getattr(PARAM, 'MIN_WORD_LEN', None)}")
                lines.append(f"  MAX_WORD_LEN: {getattr(PARAM, 'MAX_WORD_LEN', None)}")
                lines.append(f"  TOTAL_WORDS: {getattr(PARAM, 'TOTAL_WORDS', None)}")
                lines.append(f"  WORD_TYPE: {getattr(PARAM, 'WORD_TYPE', None)}")
                lines.append(f"  LEMMA_BATCH_SIZE: {getattr(PARAM, 'LEMMA_BATCH_SIZE', None)}")
                lines.append("")
                lines.append("[Selection]")
                lines.append(f"  unique targets: {n_targets}")
                lines.append(f"  unique primer2: {n_p2}")
                lines.append(f"  unique primer1: {n_p1}")
                lines.append(f"  rows_before: {rows_before}")
                lines.append(f"  rows_removed: {rows_before - n_rows_after}")
                lines.append(f"  rows_after: {n_rows_after}")
                lines.append("  dist_group_counts:")
                for g, c in sorted(dist_counts.items()):
                    lines.append(f"    group {g}: {c}")
                if words_to_delete:
                    lines.append("")
                    lines.append(f"Deleted word count: {len(words_to_delete)}")
                    lines.append(f"Deleted words: " + ", ".join(words_to_delete))
                lines.append("")
                lines.append("[Versions]")
                for k_v, v_v in vers.items():
                    lines.append(f"  {k_v}: {v_v}")

                log_path.write_text("\n".join(lines), encoding="utf-8")
                print(f"Wrote run log: {log_path}")
            except Exception as e:
                print(f"Warning: failed to write run log: {e}")

main()
