from utils import *

import re
from collections import defaultdict
from itertools import combinations

import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
import textwrap

from crosscoder import CrossCoder

torch.set_printoptions(precision=16)

def format_tokens(t):
    """Format token number to human readable form."""
    if t >= 1e9:
        return f"{t/1e9:.0f}B"
    elif t >= 1e6:
        return f"{t/1e6:.0f}M"
    else:
        return str(t)

def revision2tokens_pythia(revision):
    """
    Convert a Pythia training step to a human-readable token count.
    Each step in Pythia is 2M tokens.
    """
    if revision == "main":
        return "286B"
    
    step = int(revision.split("step")[1])
    tokens = step * 2_000_000
    
    return format_tokens(tokens)
    
    
def revision2tokens_olmo(revision):
    """
    Convert an OLMo training step to a human-readable token count.
    This is automatically added in the revision ending.
    """
    if revision == "main":
        return "3048B+cooldown"
    
    return revision.split("tokens")[1]

def revision2tokens_bloom(revision):
    """
    Convert a BLOOM training step to a human-readable token count.
    Each step in BLOOM 1B is roughly 550K tokens.
    """
    if revision == "main":
        return "341B"
    
    tokens = int(revision.split("step")[-1])
    tokens = int(tokens * 550_000)
    
    return format_tokens(tokens)

def calc_rel_dec_norm(crosscoder, one_vs_all=True):
    """
    Given crosscoder W_dec calculates RelDec norm, only supports 2 or 3 model 
    comparisons. For 3 models, if one_vs_all then calculates that RelDec norm, 
    otherwise does pairwise RelDec norms for each combo.
    """
    model_num = crosscoder.W_dec.shape[1]
    relative_norms = None
    norms = crosscoder.W_dec.norm(dim=-1)
    if model_num == 2:
        relative_norms = norms[:, 1] / norms.sum(dim=-1)
    elif model_num == 3:
        if one_vs_all:
            relative_norms = torch.stack([
                norms[:, 0] / norms.sum(dim=-1),
                norms[:, 1] / norms.sum(dim=-1),
                norms[:, 2] / norms.sum(dim=-1)
            ], dim=0)
        else:
            relative_norms = torch.stack([
                norms[:, 1] / (norms[:, 0] + norms[:, 1]),
                norms[:, 2] / (norms[:, 1] + norms[:, 2]),
                norms[:, 2] / (norms[:, 0] + norms[:, 2])
            ], dim=0)
    else:
        raise ValueError("calc_rel_dec_norm only supports 2 or 3 models in W_dec.")

    return relative_norms.detach().cpu().numpy()

def calc_rel_ie(abs_ie_array, remove_nonzero=False, one_vs_all=True):
    """
    Given absolute IE array, calculates RelIE.
    """
    total = np.sum(abs_ie_array, axis=0)
    not_all_zero = total != 0.0
    all_zero = total == 0.0
    all_zero_count = all_zero.sum().item()
    
    m1, m2, m3 = None, None, None
    if len(abs_ie_array) == 2:
        m1, m2 = abs_ie_array
    elif len(abs_ie_array) == 3:
        m1, m2, m3 = abs_ie_array
    else:
        raise ValueError("abs_ie_array must have 2 or 3 elements.")
    
    relative_ie = None
    if remove_nonzero:
        if len(abs_ie_array) == 2:
            relative_ie = m2[not_all_zero] / total[not_all_zero]
        elif len(abs_ie_array) == 3:
            if one_vs_all:
                relative_ie = np.array([
                        m1[not_all_zero] / total[not_all_zero],
                        m2[not_all_zero] / total[not_all_zero], 
                        m3[not_all_zero] / total[not_all_zero]
                    ])
            else:
                relative_ie = np.array([
                        m2[not_all_zero] / (m1 + m2)[not_all_zero],
                        m3[not_all_zero] / (m2 + m3)[not_all_zero],
                        m3[not_all_zero] / (m1 + m3)[not_all_zero],
                    ])
    else:
        # NOTE: you get warning for the following operations because it first does
        #       m2 / total, and then replaces the invalid elements with 0.5, so don't worry
        
        if len(abs_ie_array) == 2:
            relative_ie = np.where(
                all_zero,
                0.5,
                m2 / total
            )
        elif len(abs_ie_array) == 3:
            if one_vs_all:
                relative_ie = np.where(
                    np.tile(all_zero, (3, 1)),
                    0.5,
                    np.array([
                        m1 / total,
                        m2 / total,
                        m3 / total
                    ])
                )
            else:
                relative_ie = np.where(
                    np.tile(all_zero, (3, 1)),
                    0.5,
                    np.array([
                        m2 / (m1 + m2),
                        m3 / (m2 + m3),
                        m3 / (m1 + m3)
                    ])
                )
    
    return relative_ie, all_zero_count

def save_feature_vis(
    model_list, 
    folded_crosscoder, 
    latents_to_study, 
    tokens, 
    attn, 
    num_examples, 
    device, 
    filename
):
    """Save feature vis with the modified SAE vis package.

    Args:
        model_list: list of LanguageModel-s from NNSight
        folded_crosscoder: folded version of the crosscoder
        latents_to_study: a set of feature IDs whose activations we would like to visualize
        tokens: tokens over which to calculate the activations
        attn: attn mask of tokens
        num_examples: number of examples to use from tokens, 
                      the LModeling datasets can be large, so we take a subset
        device: cuda or cpu
        filename: the filename for the saved HTML
    """
    from sae_vis.model_fns import CrossCoder as SaeVisCrossCoder
    is_btopk = False
    if "batch_topk_final" in folded_crosscoder.cfg \
        and folded_crosscoder.cfg["batch_topk_final"] is not None:
            is_btopk = True
    
    sae_vis_crosscoder = SaeVisCrossCoder(cfg=folded_crosscoder.cfg, is_btopk=is_btopk)
    sae_vis_crosscoder.load_state_dict(folded_crosscoder.state_dict())
    sae_vis_crosscoder = sae_vis_crosscoder.to(device)
    sae_vis_crosscoder.eval()
    assert torch.equal(sae_vis_crosscoder.W_dec, folded_crosscoder.W_dec)
    
    from sae_vis.data_config_classes import SaeVisConfig

    sae_vis_config = SaeVisConfig(
        hook_point = folded_crosscoder.cfg["hook_point"],
        features = latents_to_study,
        verbose = True,
        minibatch_size_tokens=4,
        minibatch_size_features=16
    )
    
    from sae_vis.data_storing_fns import SaeVisData
    sae_vis_data = SaeVisData.create(
        encoder = sae_vis_crosscoder,
        encoder_B = None,
        model_list = model_list,
        tokens = tokens[:num_examples],
        attn = attn[:num_examples],
        cfg = sae_vis_config,
    )
    print(f"Saving visualization to {filename}")
    sae_vis_data.save_feature_centric_vis(filename)


def serve_html_file(filename: str, height: int = 850, port: int = 8084):
    """Serves the saved SAE vis HTML file."""
    import webbrowser
    import http.server
    import socketserver
    import threading
    import os
    import signal
    import sys

    abs_path = os.path.abspath(filename)
    directory = os.path.dirname(abs_path)
    basename = os.path.basename(abs_path)

    class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

    def serve():
        os.chdir(directory)
        with socketserver.TCPServer(("", port), NoCacheHTTPRequestHandler) as httpd:
            print(f"✅ Serving files from {directory} on port {port}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass

    server_thread = threading.Thread(target=serve)
    server_thread.daemon = True
    server_thread.start()

    url = f"http://localhost:{port}/{basename}"
    print(f"Opening {url} in browser...")
    webbrowser.open(url)

    print("Press Ctrl+C to stop the server and exit.")
    try:
        signal.pause()  # Wait for interrupt
    except KeyboardInterrupt:
        print("\nShutting down.")
        sys.exit(0)


def aggregate_eval_res(
    model_str, 
    model_name, 
    replace_string, 
    assert_num,
    joint_eval_sparsity_path="workspace/results/sparsity_ce/joint_eval_sparsity",
    joint_eval_ce_path="joint_eval_ce",
):
    """Aggregate sparsity and CE JSON results into tidy DataFrames and CSVs.

    Scans two folders for JSONs, keeps runs with seeds {124, 153, 6582} matching
    the given model family, normalizes model_type, assigns categories, adds a
    human-readable token label, and outputs:
      - ./workspace/results/sparsity_ce/joint_eval_aggregated/sparsity_{model_str}.csv
      - ./workspace/results/sparsity_ce/joint_eval_aggregated/ce_{model_str}.csv

    Args:
        model_str (str): Suffix for output filenames
        model_name (str): HF model name
        replace_string (str): Substring to strip from model_type
        assert_num (int): Expected count of included JSONs per folder (asserted)
        joint_eval_sparsity_path (str): Input dirs for sparsity JSONs.
        joint_eval_ce_path (str): Input dirs for CE JSONs.
        
    Returns: sparsity_df, ce_df : pandas.DataFrame
    """
    
    def reformat_olmo(model_type):
        """Helper: convert a raw model_type string into a normalized format for OLMO models"""
        model_type = model_type.replace("main", "main-main")
        parts = model_type.split("-")
        seps = ['-' if i % 2 == 0 else '_' for i in range(len(parts) - 1)]
        out = "".join(
            part + (seps[i] if i < len(seps) else "")
            for i, part in enumerate(parts)
        )
        model_type.replace("main-main", "main")
        return out
        
    def get_tokens(model_type):
        """Helper: map a model_type into human-readable token counts for plotting/comparison"""
        token_stage_str = ""
        if model_name == "olmo":
            try:
                token_stage = [revision2tokens_olmo(step) for step in model_type.split("_")]
                token_stage_str = " vs. ".join(token_stage)
            except:
                return ""
        elif model_name == "pythia":
            try:
                token_stage = [revision2tokens_pythia(step) for step in model_type.split("-")]
                token_stage_str = " vs. ".join(token_stage)
            except:
                return ""
        elif model_name == "bloom":
            try:
                token_stage = [revision2tokens_bloom(step) for step in model_type.split("-")]
                token_stage_str = " vs. ".join(token_stage)
            except:
                return ""
        return token_stage_str
    
    ################################################################################
    # Part 1: Load and aggregate sparsity JSON results
    ################################################################################
    rdict_list = []
    for filename in os.listdir(joint_eval_sparsity_path):
        # Skip any “list…” files and non-JSON
        if not filename.startswith("list") and filename.endswith(".json"):
            with open(os.path.join(joint_eval_sparsity_path, filename)) as f:
                rdict = json.load(f)
                if "seed" in rdict:
                    # Only include entries that match seed and model_name,
                    if model_name in rdict["model_type"].lower() and rdict["seed"] in [124, 153, 6582]:
                        rdict_list.append(rdict)

    # Sanity check: must match expected number of runs
    assert len(rdict_list) == assert_num

    # Build a DataFrame of the collected dicts
    df = pd.DataFrame(rdict_list)
    subselected_cols = [
        "model_type", "ae_dim", "seed", "data_split",
        "l0_loss-mean", "l2_loss-mean",
        "explained_variance_A-mean", "explained_variance_B-mean", "explained_variance_C-mean",
        "dead_count"
    ]
    if "explained_variance_C-mean" not in df.columns:
        subselected_cols.remove("explained_variance_C-mean")
    df = df[subselected_cols]
    # Sort rows by model_type, autoencoder dimension, and data split
    df = df.sort_values(by=['model_type', 'ae_dim', 'data_split'], ascending=[True, True, True])
    # Strip out unwanted substring from model_type
    df["model_type"] = df["model_type"].apply(lambda x: x.replace(replace_string, ""))
    # If olmo, apply the custom reformatting
    if model_name == "olmo":
        df["model_type"] = df["model_type"].apply(reformat_olmo)
    # Assign categories: either “consecutive” for all, or look up via dict
    df["categories"] = df.apply(lambda row: "consecutive", axis=1)
    # Drop any rows whose category list is empty
    df = df[df["categories"].map(lambda x: len(x) > 0)].reset_index(drop=True)
    # Generate human-readable token labels for plotting
    df["token_name"] = df.apply(lambda row: get_tokens(row["model_type"]), axis=1)
    
    # Write the DataFrame to a CSV file
    df.to_csv(f'./workspace/results/sparsity_ce/joint_eval_aggregated/sparsity_{model_str}.csv', index=False)
    
    sparsity_df = df

    ################################################################################
    # Part 2: Load and aggregate cross‐entropy (CE) JSON results
    ################################################################################
    rdict_list = []

    for filename in os.listdir(joint_eval_ce_path):
        # Skip any “list…” files and non-JSON
        if not filename.startswith("list"):
            with open(os.path.join(joint_eval_ce_path, filename)) as f:
                rdict = json.load(f)
                if "seed" in rdict:
                    if model_name in rdict["model_type"].lower() and rdict["seed"] in [124, 153, 6582]:
                        rdict_list.append(rdict)

    assert len(rdict_list) == assert_num

    df = pd.DataFrame(rdict_list)
    subselected_cols = ["model_type", "ae_dim", "seed", "data_split", 
        'ce_clean_A-mean', 'ce_loss_spliced_A-mean', 'ce_zero_abl_A-mean', 'ce_diff_A-mean', 'ce_recovered_A-mean',
        'ce_clean_B-mean', 'ce_loss_spliced_B-mean', 'ce_zero_abl_B-mean', 'ce_diff_B-mean', 'ce_recovered_B-mean',
        'ce_clean_C-mean', 'ce_loss_spliced_C-mean', 'ce_zero_abl_C-mean', 'ce_diff_C-mean', 'ce_recovered_C-mean']
    if "ce_diff_C-mean" not in df.columns:
        for c in ['ce_clean_C-mean', 'ce_loss_spliced_C-mean', 'ce_zero_abl_C-mean', 'ce_diff_C-mean', 'ce_recovered_C-mean']:
            subselected_cols.remove(c)
    df = df[subselected_cols]
    df = df.sort_values(by=['model_type', 'ae_dim', 'seed', 'data_split'], ascending=[True, True, True, True])
    df["model_type"] = df["model_type"].apply(lambda x: x.replace(replace_string, ""))
    if model_name == "olmo":
        df["model_type"] = df["model_type"].apply(reformat_olmo)
    df["categories"] = df.apply(lambda row: "consecutive", axis=1)
    df = df[df["categories"].map(lambda x: len(x) > 0)].reset_index(drop=True)
    df["token_name"] = df.apply(lambda row: get_tokens(row["model_type"]), axis=1)
    
    # Now write the DataFrame to a CSV file
    df.to_csv(f'./workspace/results/sparsity_ce/joint_eval_aggregated/ce_{model_str}.csv', index=False)
    
    ce_df = df
    return sparsity_df, ce_df


def plot_sorted_distributions(
    dist1, 
    dist2, 
    label1="Dist M1",
    label2="Dist M2", 
    xlabel="Features", 
    ylabel="IE Values", 
    title="Sorted Distributions (Filter 0.0 For Both Models)",
    bins=100,
    save_path=None
):
    # 1) Convert and sort values
    vals1, _ = torch.sort(dist1)
    vals2, _ = torch.sort(dist2)
    both_zero = torch.logical_and((vals1 == 0.0), (vals2 == 0.0))
    vals1 = vals1[~both_zero]
    vals2 = vals2[~both_zero]
    vals1 = vals1.tolist()
    vals2 = vals2.tolist()
        
    # 2) Plot histograms
    plt.figure(figsize=(10, 5))
    plt.hist(vals1, bins=bins, alpha=0.6, label=label1, edgecolor='black')
    plt.hist(vals2, bins=bins, alpha=0.6, label=label2, edgecolor='black')
    
    # 3) Labeling
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # 4) Save if needed
    if save_path:
        plt.savefig(save_path)

    plt.show()


def rel_ie_histogram(
    m1_effects, 
    m2_effects, 
    label1="M1", 
    label2="M2", 
    title="Relative Non-Zero IE Histogram", 
    save_path=None,
    show_fig=False
):
    relative_ie, both_zero_count = calc_rel_ie(
        m1_effects, 
        m2_effects,
        remove_nonzero=True
    )

    df = pd.DataFrame({"Relative Non-Zero Abs IE": relative_ie})

    stats_text = f"""
    <b>Relative Non-Zero Abs IE Stats:</b><br>
    Min: {df["Relative Non-Zero Abs IE"].min():.4f}<br>
    Max: {df["Relative Non-Zero Abs IE"].max():.4f}<br>
    Mean: {df["Relative Non-Zero Abs IE"].mean():.4f}<br>
    Median: {df["Relative Non-Zero Abs IE"].median():.4f}<br>
    Std Dev: {df["Relative Non-Zero Abs IE"].std():.4f}<br>
    <br>
    # of Zero for Both Models: {both_zero_count}<br>
    """

    fig = px.histogram(
        df,
        x="Relative Non-Zero Abs IE",
        nbins=100,
        title=f"{title} ({label2} / [{label1} + {label2}])",
        labels={"Relative Non-Zero Abs IE": "Relative Non-Zero Abs IE"},
        opacity=0.75
    )

    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    fig.update_yaxes(title_text="Number of Latents")

    fig.update_layout(
        width=1000,
        height=600,
        margin=dict(l=80, r=240, t=80, b=80),
        showlegend=False
    )

    fig.add_annotation(
        xanchor="left", yanchor="top",
        x=1.01, y=0.99, xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        opacity=0.8
    )

    if save_path:
        fig.write_image(save_path)
        
    if show_fig:
        fig.show()


def ie_rel_dec_confusion_matrix(df, ie_column, top_k, save_path=None):
    from sklearn.metrics import confusion_matrix
    
    assert len(df['task'].unique()) == 1
    task = df['task'].unique()[0]
    valid_values = {0, 0.5, 1.0}

    # 1) Check for NaNs
    assert df[ie_column].notna().all(), f"Found NaNs in {ie_column}"
    assert df["rel_dec_norm_class"].notna().all(), "Found NaNs in rel_dec_norm_class"

    # 2) Check for invalid values
    invalid_ie_vals = set(df[ie_column].unique()) - valid_values
    invalid_reldec_vals = set(df["rel_dec_norm_class"].unique()) - valid_values

    assert not invalid_ie_vals, f"Invalid values in {ie_column}: {invalid_ie_vals}"
    assert not invalid_reldec_vals, f"Invalid values in rel_dec_norm_class: {invalid_reldec_vals}"
    
    labels = ["0.0", "0.5", "1.0"]
    y_true = df[ie_column].astype(str)
    y_pred = df["rel_dec_norm_class"].astype(str)
    
    # 3) Check for NaNs again
    assert y_true.notna().all(), f"Found NaNs in {ie_column}"
    assert y_pred.notna().all(), "Found NaNs in rel_dec_norm_class"
    
    print("y_true unique", y_true.unique())
    print("y_pred unique", y_pred.unique())
    print("labels", labels)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    cm_df = pd.DataFrame(cm, index=[f"IE class: {l}" for l in labels],
                             columns=[f"RelDec class: {l}" for l in labels])

    # 4) Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Purples", cbar=False)
    ie_name = "TopK IE" if ie_column == "topk_ie_class" else "RelIE"
    plt.title(f"{task} - top_k={top_k} - {ie_name} vs RelDec Classification Comparison")
    plt.ylabel("IE Classification")
    plt.xlabel("RelDec Classification")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    return cm_df

def relative_corr(df, save_path=None):
    task = df['task'].unique()[0]
    rel_ie_value = df["rel_ie_value"]
    rel_dec_value = df["rel_dec_norm_value"]

    plt.figure(figsize=(8, 6))
    plt.scatter(rel_ie_value, rel_dec_value, alpha=0.7)
    plt.xlabel("Relative IE")
    plt.ylabel("Relative Decoder Norm")
    plt.title(f"{task} - Scatter Plot of RelIE vs RelDecNorm")
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_top_bottom_evolution(
    version_num   = 447,
    max_examples  = 3469,
    node_threshold= 0.1,
    task          = "subjectverb",
    ckpt_num      = 20,
    top_k         = 5,
    ckpt_tokens   = ['1B', '4B', '286B'],
    figsize       = (7,6),
    save_path     = None,
    fontsize      = 22,
    markersize    = 7
):
    base_dir   = "./workspace/logs/ie_dicts_zeroshot"
    save_dir   = f"{base_dir}/version_{version_num}"
    final_path = f"{save_dir}/{task}_ckpt{ckpt_num}_thresh{node_threshold}_n{max_examples}.pt"
    effects    = torch.load(final_path)

    ckpt_keys = ["m0_layer8_out", "m1_layer8_out", "m2_layer8_out"]

    # 1) Load descriptions
    desc_df  = pd.read_csv(f"{save_dir}/top5_bottom5_annotation.csv")
    desc_map = dict(zip(desc_df["feat_id"], desc_df["description"]))

    # 2) Pick top/bottom features
    _, bb = effects[ckpt_keys[0]].topk(top_k, largest=False)
    _, tb = effects[ckpt_keys[0]].topk(top_k, largest=True)
    _, be = effects[ckpt_keys[-1]].topk(top_k, largest=False)
    _, te = effects[ckpt_keys[-1]].topk(top_k, largest=True)

    # 3) Define your groups, including any overlap‐combos you want
    groups = {
        "bottom_begin":        bb.tolist(),
        "top_begin":           tb.tolist(),
        "bottom_end":          be.tolist(),
        "top_end":             te.tolist(),
    }
    base_keys = ["bottom_begin","top_begin","bottom_end","top_end"]
    for g1, g2 in combinations(base_keys, 2):
        combo_key = f"{g1}&{g2}"
        # compute intersection of feature‑IDs
        shared = set(groups[g1]).intersection(groups[g2])
        if shared:
            groups[combo_key] = list(shared)

    # 4) Color + marker for *each* group‑key
    color_map = {
        "bottom_begin":              "#B6A74D",
        "top_begin":                 "#4B8948",
        "bottom_end":                "#BA5FAE",
        "top_end":                   "#512DA8",
        "bottom_begin&top_begin":    "#FF8C00",
        "bottom_begin&bottom_end":   "#8B4513",
        "bottom_begin&top_end":      "#006400",
        "top_begin&bottom_end":      "#6495ED",
        "top_begin&top_end":         "#00B0F6",
        "bottom_end&top_end":        "#DC143C",
    }
    marker_map = {
        "bottom_begin":            "o",
        "top_begin":               "o",
        "bottom_end":              "s",
        "top_end":                 "s",
        "bottom_begin&top_begin":  "X",
        "bottom_begin&bottom_end": "X",
        "bottom_begin&top_end":    "X",
        "top_begin&bottom_end":    "X",
        "top_begin&top_end":       "X",
        "bottom_end&top_end":      "X",
    }

    # 5) Helper: extract IE series
    def series(fid_list):
        return {fid: [effects[k][fid].item() for k in ckpt_keys] for fid in fid_list}
    data = {name: series(ids) for name, ids in groups.items()}

    # 6) Build fid→group mapping, so each feature “knows” its plotted group
    #    We do singles first, then combos override
    fid_to_group = {}
    for name, fids in groups.items():
        if '&' not in name:
            for fid in fids:
                fid_to_group[fid] = name
    for name, fids in groups.items():
        if '&' in name:
            for fid in fids:
                fid_to_group[fid] = name

    # 7) Helper to make the “pretty” legend label from the group key
    def make_label(name):
        if '&' in name:
            parts = name.split('&')
            # parse each part into (prefix, pos) → (token, category)
            tokens, cats = [], []
            for part in parts:
                prefix, pos = part.split('_')
                tokens.append(ckpt_tokens[0] if pos=='begin' else ckpt_tokens[-1])
                cats.append(prefix.capitalize())
            # if same category, join tokens, else join token+cat pairs
            if cats[0] == cats[1]:
                label = f"{tokens[0]} & {tokens[1]} {cats[0]}"
                if name == "bottom_begin&bottom_end":
                    # Insert newline *before* the category (i.e. before 'Bottom')
                    label = label.replace(f" {cats[0]}", f"\n{cats[0]}")
                return label
            else:
                return " & ".join(f"{tok} {cat}" for tok,cat in zip(tokens,cats))
        else:
            prefix, pos = name.split('_')
            cat   = "Bottom" if prefix=='bottom' else "Top"
            token = ckpt_tokens[0] if pos=='begin' else ckpt_tokens[-1]
            return f"{token} {cat}"

    # 8) Plot everything in one loop
    fig, ax = plt.subplots(figsize=figsize)
    for name, fids in groups.items():
        clr = color_map[name]
        mkr = marker_map[name]
        lbl = make_label(name)

        first = True
        for fid in fids:
            # only plot this fid in exactly one group
            if fid_to_group[fid] != name:
                continue

            ax.plot(
                [1,2,3],
                data[name][fid],
                color=clr,
                marker=mkr,
                markersize=markersize,
                alpha=1.0,
                label=lbl if first else None
            )
            first = False

    # 9) Axes styling
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(ckpt_tokens, fontsize=fontsize)
    ax.set_xlabel('Checkpoint', fontsize=fontsize)
    ax.set_ylabel('IE value', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_title(
        'Top & Bottom Features\' IE value over Time', 
        fontsize=fontsize - 2, 
        pad=20
    )

    # 10) Build legend: singles first, combos last, text colored to match each curve
    handles, labels = ax.get_legend_handles_labels()
    singles = [(lab,h) for lab,h in zip(labels,handles) if '&' not in lab]
    combos  = [(lab,h) for lab,h in zip(labels,handles) if '&' in lab]
    singles.sort(key=lambda x: x[0].lower())
    combos .sort(key=lambda x: x[0].lower())
    ordered = singles + combos
    labs, hnds = zip(*ordered)

    # map legend‐label to its curve‐color
    label_to_color = { make_label(k): v for k,v in color_map.items() }
    lbl_colors = [ label_to_color[lab] for lab in labs ]

    ax.legend(
        hnds, labs,
        fontsize=fontsize - 1,
        markerscale=1.5,
        labelcolor=lbl_colors,
        ncol=1,
        loc='best',
        framealpha=0.5
    )

    # 11) Annotate, using fid_to_group for text color
    unique_fids = list(fid_to_group.keys())
    ys_final = {fid: effects[ckpt_keys[-1]][fid].item() for fid in unique_fids}

    # 12) Filter out uninterpretable features early
    annotatable_fids = [fid for fid in unique_fids if desc_map.get(fid, "").strip() not in ["", "-", "N/A"]]
    sorted_fids = sorted(annotatable_fids, key=lambda f: ys_final[f], reverse=True)
    sorted_y_act = [ys_final[f] for f in sorted_fids]

    # 13) Recalculate y-range *only* for valid annotations
    n = len(sorted_fids)
    y_min, y_max = ax.get_ylim()
    y_center, y_range = (y_max + y_min) / 2, y_max - y_min
    stretch_factor = 0.95
    y_label_sorted = np.linspace(
        y_center + (y_range * stretch_factor) / 2,
        y_center - (y_range * stretch_factor) / 2,
        n
    )

    # 15) Annotate only valid ones
    for fid, y_act, y_lbl in zip(sorted_fids, sorted_y_act, y_label_sorted):
        desc = desc_map.get(fid, "").replace("\\n", "\n")
        txt = f"#{fid}  {desc}"

        grp = fid_to_group[fid]
        ann_color = color_map[grp]

        ax.annotate(
            "",
            xy=(3, y_act),
            xytext=(3.4, y_lbl),
            arrowprops=dict(
                arrowstyle='<-',
                shrinkA=0, shrinkB=0,
                lw=1.0, color='darkgrey',
                mutation_scale=20
            ),
        )
        ax.text(
            3.4, y_lbl, txt,
            ha='left', va='center',
            fontsize=fontsize,
            multialignment='left',
            color=ann_color,
            weight=550,
            linespacing=0.9
        )

    # 16) Save the figure
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    else:
        plt.show()


def table_generator_three_way(
    version_num    = 446,
    model_name     = "Pythia-1B"
):
    base_dir   = "./workspace/logs/ie_dicts_zeroshot"
    save_dir   = f"{base_dir}/version_{version_num}"
    final_path = f"{save_dir}/annotation.csv"

    # 1. Read & subset
    df = pd.read_csv(final_path)
    df = df[['comparison', 'rel_ie_value', 'feat_id', 'description']].copy()

    # 2. Extract model names
    comp_raw = df['comparison'].unique()[0]
    models   = re.split(r'\s*vs\.\s*', comp_raw.strip())
    modelA, modelB, modelC = models

    # 3. Parse & classify
    def parse_vec(s):
        parts = [p for p in s.strip('[]').split() if p]
        return [float(x) for x in parts]
    df['vec'] = df['rel_ie_value'].apply(parse_vec)

    def classify(v):
        a, b, c = v
        if a > 0.7 and a > b and a > c:               return 'A dominant'
        if b > 0.7 and b > a and b > c:               return 'B dominant'
        if c > 0.7 and c > a and c > b:               return 'C dominant'
        if a >= 0.3 and b >= 0.3 and c >= 0.3:         return 'A-B-C shared'
        if a >= 0.3 and b >= 0.3:                     return 'A-B shared'
        if a >= 0.3 and c >= 0.3:                     return 'A-C shared'
        if b >= 0.3 and c >= 0.3:                     return 'B-C shared'
        return 'Other'
    df['orig_cat'] = df['vec'].apply(classify)

    # 4. Map to final group labels using checkpoint names
    def map_group(cat):
        if   cat == 'A dominant':     return f'{modelA} specific'
        elif cat == 'A-B shared':     return f'{modelA}-{modelB} shared'
        elif cat == 'A-C shared':     return f'{modelA}-{modelC} shared'
        elif cat == 'B dominant':     return f'{modelB} specific'
        elif cat == 'B-C shared':     return f'{modelB}-{modelC} shared'
        elif cat == 'C dominant':     return f'{modelC} specific'
        elif cat == 'A-B-C shared':   return f'{modelA}-{modelB}-{modelC} shared'
        else:                         return 'Others'
    df['group'] = df['orig_cat'].apply(map_group)

    # 5. Prepare printed columns
    df['comparison'] = df['comparison'].str.replace('vs.', r'\compar{}', regex=False)
    def round_vec_math(s):
        parts = [p for p in s.strip('[]').split() if p]
        coords = [float(x) for x in parts]
        return '$[' + ',\\;'.join(f"{c:.2f}" for c in coords) + ']$'
    df['rel_ie_value'] = df['rel_ie_value'].apply(round_vec_math)

    # 6. Caption’s comparison
    comp_vals = df['comparison'].unique()
    comp_str  = comp_vals[0] if len(comp_vals) == 1 else r'\text{Multiple}'

    # 7. Drop helpers
    df = df.drop(columns=['comparison', 'vec', 'orig_cat'])

    # 8. Rename headers
    df = df.rename(columns={
        'rel_ie_value': r'\mythead{\relie{}}',
        'feat_id':      r'\mythead{FeatID}',
        'description':  r'\mythead{Interpreted Function}'
    })

    # 9. Sort by your group order (using checkpoint names)
    final_order = [
        f'{modelA} specific',
        f'{modelA}-{modelB} shared',
        f'{modelA}-{modelC} shared',
        f'{modelB} specific',
        f'{modelB}-{modelC} shared',
        f'{modelC} specific',
        f'{modelA}-{modelB}-{modelC} shared',
        'Others'
    ]
    df['group'] = pd.Categorical(df['group'], categories=final_order, ordered=True)
    df = df.sort_values('group')

    # 10. Emit LaTeX
    hdrs = list(df.columns.drop('group'))
    lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\resizebox{1.0\\linewidth}{!}{%",
        "\\begin{tabular}{crl}",
        "\\toprule",
        " & ".join(hdrs) + r" \\",
        "\\midrule",
    ]
    for grp, block in df.groupby('group', sort=False):
        if block.empty:
            continue
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\bfseries {grp}}} \\\\")
        lines.append("\\midrule")
        for _, row in block.iterrows():
            vals = [str(row[c]) for c in hdrs]
            lines.append(" & ".join(vals) + r" \\")

    caption = "".join([
        "\\caption{",
        f"\\textbf{{3-way L1‑Sparsity Crosscoder Annotation for {model_name} | Comparison {comp_str}.}} ",
        "\\relie{} shows 3-way one-vs-all \\relie{} vector; ",
        "Interpreted Function provides a description if a linguistic role was detected, and ``--'' otherwise. ",
        "Rows are grouped by checkpoint specificity: "
        f"features dominated by one checkpoint ({modelA}, {modelB}, {modelC} specific); "
        f"pairwise shared features ({modelA}–{modelB}, {modelA}–{modelC}, {modelB}–{modelC} shared); "
        f"and shared across all ({modelA}–{modelB}–{modelC} shared). ",
        "A missing group means no such features found in the top-10 IE features of all checkpoints.",
        "}"
    ])
    
    model_name_formatted = model_name.split("-")[0].lower()
    
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        caption,
        f"\\label{{tab:3way_annotation_{model_name_formatted}}}",
        "\\end{table*}"
    ]

    return "\n".join(lines)


def table_generator_two_way(
    annotation_filename,
    model_name="Pythia-1B"
):
    # 1. Load file & subset
    df = pd.read_csv(annotation_filename, usecols=[
        'comparison','rel_ie_value','feat_id','description'
    ]).copy()

    # 2. Parse out checkpoint names
    def split_ab(comp):
        a, b = re.split(r'\s*vs\.\s*', comp.strip())
        return a, b
    df[['modelA','modelB']] = df['comparison'].apply(
        lambda s: pd.Series(split_ab(s))
    )

    # 3. Numeric RelIE + classify into 3 bins
    df['rel_ie_num'] = df['rel_ie_value'].astype(float)
    def classify(v):
        if v < 0.3:    return 'A specific'
        if v > 0.7:    return 'B specific'
        return 'A-B shared'
    df['orig_cat'] = df['rel_ie_num'].apply(classify)

    # 4. Map each row to its human‐readable subgroup
    def map_group(r):
        if   r.orig_cat=='A specific':   return f"{r.modelA} specific"
        elif r.orig_cat=='A-B shared':   return f"{r.modelA}-{r.modelB} shared"
        elif r.orig_cat=='B specific':   return f"{r.modelB} specific"
        else:                             return 'Others'
    df['group'] = df.apply(map_group, axis=1)

    # 5. Prepare LaTeX‐formatted columns
    df['rel_ie_value'] = df['rel_ie_num'].apply(lambda x: f"${x:.2f}$")
    df['comp_ltx']     = df['comparison'].str.replace('vs.', r'\compar{}', regex=False)

    # 6. Rename to your \mythead headers
    df = df.rename(columns={
        'rel_ie_value':       r'\mythead{\relie{}}',
        'feat_id':            r'\mythead{FeatID}',
        'description':        r'\mythead{Interpreted Function}'
    })

    # 7. Build the LaTeX
    hdrs = [
        r'\mythead{\relie{}}',
        r'\mythead{FeatID}',
        r'\mythead{Interpreted Function}',
    ]
    lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\resizebox{1.0\\linewidth}{!}{%",
        # now 3 columns: RelIE | FeatID | Interpreted Function
        "\\begin{tabular}{crl}",
        "\\toprule",
        " & ".join(hdrs) + r" \\",
        "\\midrule",
    ]

    # preserve the order of comparisons as they appear
    comp_order = (
        df[['comparison','comp_ltx']]
        .drop_duplicates('comparison')
        .set_index('comparison')['comp_ltx']
        .to_dict()
    )

    for comp_raw, comp_ltx in comp_order.items():
        sub = df[df['comparison']==comp_raw].copy()

        # define this block's local subgroup ordering
        a,b = split_ab(comp_raw)
        local_groups = [
            f"{a} specific",
            f"{a}-{b} shared",
            f"{b} specific",
            "Others",
        ]
        sub['group'] = pd.Categorical(sub['group'],
                                      categories=local_groups,
                                      ordered=True)

        # sort within this comparison by (group, then rel_ie_num)
        sub = sub.sort_values(['group','rel_ie_num'])

        # block header
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\bfseries Comparison: {comp_ltx}}} \\\\")
        lines.append("\\midrule")

        # subgroup headings + rows
        for grp, block in sub.groupby('group', sort=False):
            if block.empty:
                continue
            lines.append(f"\\multicolumn{{3}}{{l}}{{\\itshape {grp}}} \\\\")
            lines.append("\\midrule")
            for _, row in block.iterrows():
                vals = [str(row[c]) for c in hdrs]
                lines.append(" & ".join(vals) + r" \\")
        lines.append("\\addlinespace")

    # 8) Caption & closing
    caption = (
        "\\caption{\\textbf{2‑way L1‑Sparsity Crosscoder Annotation for "
        f"{model_name}."
        "} Each block is one pairwise “Comparison.” "
        "\\relie{} is sorted 0.00 to 1.00 (<0.3 $\Rightarrow$ first checkpoint; >0.7 $\Rightarrow$ second; shared otherwise). "
        "Interpreted Function gives a description if a linguistic role was detected, “--” otherwise.}"
    )
    model_low = model_name.split("-")[0].lower()
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        caption,
        f"\\label{{tab:2way_multi_{model_low}}}",
        "\\end{table*}"
    ]

    return "\n".join(lines)



def table_generator_three_way_bloom(
    annotation_filename,
    version_num    = 454,
    model_name     = "BLOOM-1B"
):
    base_dir   = "./workspace/logs/ie_dicts_zeroshot"
    save_dir   = f"{base_dir}/version_{version_num}"
    final_path = f"{save_dir}/{annotation_filename}"

    annotated_language = None
    lang_formatted = None
    if "fraeng" in annotation_filename:
        annotated_language = "French/English"
        lang_formatted = "fraeng"
    elif "fra" in annotation_filename:
        annotated_language = "French"
        lang_formatted = "fra"
    elif "eng" in annotation_filename:
        annotated_language = "English"
        lang_formatted = "eng"
    elif "hin" in annotation_filename:
        annotated_language = "Hindi"
        lang_formatted = "hin"
    else:
        annotated_language = "Unknown"
        lang_formatted = "unk"
    
    task = None
    task_formatted = None
    if "multiblimp" in annotation_filename:
        task = "MultiBLiMP"
        task_formatted = "multiblimp"
    elif "clams" in annotation_filename:
        task = "CLAMS"
        task_formatted = "clams"
    else:
        task = "Unknown"
        task_formatted = "unk"

    # 1. Read & subset (now including languages)
    df = pd.read_csv(final_path)
    df = df[['comparison', 'rel_ie_value', 'feat_id', 'description', 'languages']].copy()
    
    # 1.1 Replace half-space characters in descriptions with a normal space
    df['description'] = df['description'].str.replace('\u202F', ' ', regex=False)
    
    # 1.2 Format 'languages' column: sort alphabetically, comma-separated, '-' if empty
    def format_languages(cell):
        if pd.isna(cell) or cell == "-":
            return '-'
        # Split on commas or semicolons, strip whitespace
        parts = [lang.strip() for lang in re.split(r'[;,]', str(cell)) if lang.strip()]
        # Alphabetical order
        parts_sorted = sorted(parts)
        return ','.join(parts_sorted) or '-'  
    df['languages'] = df['languages'].apply(format_languages)

    # 2. Extract model names
    comp_raw = df['comparison'].unique()[0]
    models   = re.split(r'\s*vs\.\s*', comp_raw.strip())
    modelA, modelB, modelC = models

    # 3. Parse & classify
    def parse_vec(s):
        parts = [p for p in s.strip('[]').split() if p]
        return [float(x) for x in parts]
    df['vec'] = df['rel_ie_value'].apply(parse_vec)

    def classify(v):
        a, b, c = v
        if a > 0.7 and a > b and a > c:               return 'A dominant'
        if b > 0.7 and b > a and b > c:               return 'B dominant'
        if c > 0.7 and c > a and c > b:               return 'C dominant'
        if a >= 0.3 and b >= 0.3 and c >= 0.3:         return 'A-B-C shared'
        if a >= 0.3 and b >= 0.3:                     return 'A-B shared'
        if a >= 0.3 and c >= 0.3:                     return 'A-C shared'
        if b >= 0.3 and c >= 0.3:                     return 'B-C shared'
        return 'Other'
    df['orig_cat'] = df['vec'].apply(classify)

    # 4. Map to final group labels using checkpoint names
    def map_group(cat):
        if   cat == 'A dominant':     return f'{modelA} specific'
        elif cat == 'A-B shared':     return f'{modelA}-{modelB} shared'
        elif cat == 'A-C shared':     return f'{modelA}-{modelC} shared'
        elif cat == 'B dominant':     return f'{modelB} specific'
        elif cat == 'B-C shared':     return f'{modelB}-{modelC} shared'
        elif cat == 'C dominant':     return f'{modelC} specific'
        elif cat == 'A-B-C shared':   return f'{modelA}-{modelB}-{modelC} shared'
        else:                         return 'Others'
    df['group'] = df['orig_cat'].apply(map_group)

    # 5. Prepare printed columns
    df['comparison'] = df['comparison'].str.replace('vs.', r'\compar{}', regex=False)
    def round_vec_math(s):
        parts = [p for p in s.strip('[]').split() if p]
        coords = [float(x) for x in parts]
        return '$[' + ',\\;'.join(f"{c:.2f}" for c in coords) + ']$'
    df['rel_ie_value'] = df['rel_ie_value'].apply(round_vec_math)

    # 6. Caption’s comparison
    comp_vals = df['comparison'].unique()
    comp_str  = comp_vals[0] if len(comp_vals) == 1 else r'\text{Multiple}'

    # 7. Drop helpers
    df = df.drop(columns=['comparison', 'vec', 'orig_cat'])

    # 8. Rename headers (including languages)
    df = df.rename(columns={
        'rel_ie_value': r'\mythead{\relie{}}',
        'feat_id':      r'\mythead{FeatID}',
        'description':  r'\mythead{Interpreted Function}',
        'languages':    r'\mythead{Languages}'
    })

    # 9. Sort by your group order (using checkpoint names)
    final_order = [
        f'{modelA} specific',
        f'{modelA}-{modelB} shared',
        f'{modelA}-{modelC} shared',
        f'{modelB} specific',
        f'{modelB}-{modelC} shared',
        f'{modelC} specific',
        f'{modelA}-{modelB}-{modelC} shared',
        'Others'
    ]
    df['group'] = pd.Categorical(df['group'], categories=final_order, ordered=True)
    df = df.sort_values('group')

    # 10. Emit LaTeX with 4 columns and language in title
    hdrs = list(df.columns.drop('group'))
    lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\resizebox{1.0\\linewidth}{!}{%",
        "\\begin{tabular}{crll}",
        "\\toprule",
        " & ".join(hdrs) + " \\\\",
        "\\midrule",
    ]
    for grp, block in df.groupby('group', sort=False):
        if block.empty:
            continue
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\bfseries {grp}}} \\\\")
        lines.append("\\midrule")
        for _, row in block.iterrows():
            vals = [str(row[c]) for c in hdrs]
            lines.append(" & ".join(vals) + r" \\")

    caption = "".join([
        "\\caption{",
        f"\\textbf{{3-way L1‑Sparsity Crosscoder {task} {annotated_language} Annotation for {model_name}  | Comparison {comp_str}.}} ",
        "\\relie{} shows 3-way one-vs-all \\relie{} vector; ",
        "Interpreted Function provides a description if a linguistic role was detected, and ``--'' otherwise. ",
        "Languages indicates which languages the feature is used for. ",
        "Rows are grouped by checkpoint specificity: ",
        f"features dominated by one checkpoint ({modelA}, {modelB}, {modelC} specific); ",
        f"pairwise shared features ({modelA}–{modelB}, {modelA}–{modelC}, {modelB}–{modelC} shared); ",
        f"and shared across all ({modelA}–{modelB}–{modelC} shared). ",
        "A missing group means no such features found in the top-10 IE features of all checkpoints.",
        "}"
    ])
    
    model_name_formatted = model_name.split("-")[0].lower()
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        caption,
        f"\\label{{tab:3way_annotation_{model_name_formatted}_{task_formatted}_{lang_formatted}}}",
        "\\end{table*}"
    ]

    return "\n".join(lines)



def table_generator_two_way_bloom(
    annotation_filename,
    model_name="BLOOM-1B"
):
    # 1. Read & subset
    df = pd.read_csv(annotation_filename, usecols=[
        'comparison','rel_ie_value','feat_id','description','languages'
    ])

    # 2. Clean & normalize text
    df['description'] = df['description'].str.replace('\u202F',' ', regex=False)
    def format_langs(cell):
        if pd.isna(cell) or cell == "-": return '-'
        parts = [x.strip() for x in re.split(r'[;,]', str(cell)) if x.strip()]
        return ','.join(sorted(parts)) or '-'
    df['languages'] = df['languages'].apply(format_langs)

    # 3. Split out modelA, modelB
    def split_ab(comp):
        a,b = re.split(r'\s*vs\.\s*', comp.strip())
        return a, b
    df[['modelA','modelB']] = df['comparison'].apply(
        lambda s: pd.Series(split_ab(s))
    )

    # 4. Numeric RelIE and category
    df['rel_ie_num'] = df['rel_ie_value'].astype(float)
    def classify(v):
        if v < 0.3:      return 'A specific'
        if v > 0.7:      return 'B specific'
        return 'A-B shared'
    df['orig_cat'] = df['rel_ie_num'].apply(classify)

    # 5. Map to group labels
    def map_group(r):
        if   r.orig_cat=='A specific':   return f"{r.modelA} specific"
        elif r.orig_cat=='A-B shared':   return f"{r.modelA}-{r.modelB} shared"
        elif r.orig_cat=='B specific':   return f"{r.modelB} specific"
        else:                             return 'Others'
    df['group'] = df.apply(map_group, axis=1)

    # 6. Format for LaTeX
    df['rel_ie_value'] = df['rel_ie_num'].apply(lambda x: f"${x:.2f}$")
    # Also store the LaTeX‐friendly comparison for the block headings
    df['comp_ltx'] = df['comparison'].str.replace('vs.', r'\compar{}', regex=False)

    # 7. Rename for the **four** printed columns
    df = df.rename(columns={
        'rel_ie_value': r'\mythead{\relie{}}',
        'feat_id':      r'\mythead{FeatID}',
        'description':  r'\mythead{Interpreted Function}',
        'languages':    r'\mythead{Languages}'
    })

    # 8. Prepare header list and sorting
    hdrs = [
        r'\mythead{\relie{}}',
        r'\mythead{FeatID}',
        r'\mythead{Interpreted Function}',
        r'\mythead{Languages}'
    ]

    lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\resizebox{1.0\\linewidth}{!}{%",
        # now 4 columns: RelIE | FeatID | Interpreted Function | Languages
        "\\begin{tabular}{crll}",
        "\\toprule",
        " & ".join(hdrs) + " \\\\",
        "\\midrule",
    ]

    # 9. Loop by comparison block
    # preserve original order of comparisons
    comp_order = df[['comparison','comp_ltx']] \
                  .drop_duplicates('comparison') \
                  .set_index('comparison')['comp_ltx'] \
                  .to_dict()

    for comp_raw, comp_ltx in comp_order.items():
        sub = df[df['comparison']==comp_raw].copy()

        # define this block's subgroup ordering
        a,b = split_ab(comp_raw)
        local_groups = [
            f"{a} specific",
            f"{a}-{b} shared",
            f"{b} specific",
            "Others",
        ]
        sub['group'] = pd.Categorical(sub['group'],
                                      categories=local_groups,
                                      ordered=True)

        # sort within block by group, then numeric RelIE
        sub = sub.sort_values(['group','rel_ie_num'])

        # 9a. Print comparison heading
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\bfseries Comparison: {comp_ltx}}} \\\\")
        lines.append("\\midrule")

        # 9b. Print each subgroup
        for grp, block in sub.groupby('group', sort=False):
            if block.empty:
                continue
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\itshape {grp}}} \\\\")
            lines.append("\\midrule")
            for _, row in block.iterrows():
                vals = [str(row[c]) for c in hdrs]
                lines.append(" & ".join(vals) + r" \\")
        lines.append("\\addlinespace")

    # 10. Caption & end
    caption = (
        "\\caption{\\textbf{2-way L1-Sparsity Crosscoder CLAMS French/English "
        f"Annotation for {model_name}."
        "} Each “Comparison” block shows one pair of "
        "checkpoints. Within each, \\relie{} is sorted from 0.00→1.00 "
        "(<0.3 $\Rightarrow$ first checkpoint; >0.7 $\Rightarrow$ second; otherwise shared). "
        "Interpreted Function gives a linguistic role if detected (“--” otherwise), "
        "and Languages lists which languages the feature appears in.}"
    )
    model_low = model_name.split("-")[0].lower()
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        caption,
        f"\\label{{tab:2way_multi_{model_low}_clams_fraeng}}",
        "\\end{table*}"
    ]

    return "\n".join(lines)
