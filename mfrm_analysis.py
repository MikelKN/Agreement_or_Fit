from datetime import datetime
from pathlib import Path
import os, sys, math
import RaschPy as rp
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config
print(f"Config imported successfully!")

from config import DATA_DIR, DIMENSIONS, COLORS, LLM_NAMES, ANNOTATOR_ID_MAP

# DIMENSIONS_9 = [d for d in DIMENSIONS if d != 'hatespeech']
# assert len(DIMENSIONS_9) == 9

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def read_filepaths(base_dir=DATA_DIR):
    if not base_dir.exists():
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {base_dir}")
    file_map = {}
    for f in base_dir.glob("*.csv"):
        name = f.name.lower() #lowercase everything
        if "gold" in name:
            file_map["gold"] = f
        elif "gpt5" in name and "mono" in name \
                and "mini" not in name:
            file_map["GPT5_mono"] = f

        #each llm combines with humna
        elif "human_claude_modu" in name:
            file_map["human_claude_modu"] = f
        elif "human_deepseek_modu" in name:
            file_map["human_deepseek_modu"] = f
        elif "human_gemini_modu" in name:
            file_map["human_gemini_modu"] = f
        elif "human_gpt-5-mini_modu" in name:
            file_map["human_gpt-5-mini_modu"] = f

        elif "gpt-5-mini" in name and "modu" in name:
            file_map["gpt-5-mini_modu"] = f

        elif "claude" in name and "modu" in name:
            file_map["claude_modu"] = f

        elif "gemini" in name and "modu" in name:
            file_map["gemini_modu"] = f

        elif "deepseek" in name and "modu" in name:
            file_map["deepseek_modu"] = f

        elif "llm_human_modu" in name:
            file_map["llm_human_modu"] = f

        elif 'llm_only_modu' in name:
            file_map['llm_only_modu'] = f
    return file_map

# EXPERIMENT 1: Extract a Single run from the 5 run

def extract_single_run(df, run_number=1):
    """
    Extract ratings from one specific run.
    Uses vote_distributions to get
    the first run's rating.
    """
    import json

    single_run_ratings = []
    for _, row in df.iterrows():
        try:
            vote_dist = json.loads(
                row['vote_distributions'])
        except:
            continue

        record = {
            'comment_id':   row['comment_id'],
            'annotator_id': row['annotator_id']
        }

        for dim in DIMENSIONS[:-1]:  # Exclude 'hatespeech'
            if dim not in vote_dist:
                record[dim] = None
                continue

            # Get all ratings for this dim
            ratings = []
            for score, count in vote_dist[
                    dim].items():
                ratings.extend(
                    [int(score)] * int(count))

            # Take first run's rating
            record[dim] = ratings[0] \
                if ratings else None

        single_run_ratings.append(record)

    return pd.DataFrame(single_run_ratings)

def filter_extreme_raters(df, dimensions=DIMENSIONS[:-1], 
                           min_variance=0.01):
    """
    Remove raters whose ratings have zero or near-zero variance
    across all dimensions — these cause RaschPy fit statistic errors.
    """
    rater_variances = df.groupby('annotator_id')[dimensions].var().mean(axis=1)
    extreme_raters  = rater_variances[rater_variances < min_variance].index.tolist()
    
    if extreme_raters:
        print(f"⚠️ Removing {len(extreme_raters)} extreme raters: {extreme_raters}")
        df = df[~df['annotator_id'].isin(extreme_raters)].copy()
    else:
        print("No extreme raters found")
    
    return df

def data_differentiate_raters_3(df, annotator_label=None):
    rater_params = df.copy()
    if annotator_label == 'gold':
        pass
    else:
        rater_params['annotator_id'] = annotator_label
    return rater_params


def creating_mfrm_ready_data(filepath_gold,
                            filepath_gpt5mini_modu,
                            filepath_claude_modu,
                            filepath_gemini_modu,
                            filepath_deepseek_modu,
                            output_dir):

    os.makedirs(output_dir, exist_ok=True)

    dataset_dict = {
        'gold': filepath_gold, 
        'gpt-5-mini_modu': filepath_gpt5mini_modu,
        'claude_modu': filepath_claude_modu,
        'gemini_modu': filepath_gemini_modu, 
        'deepseek_modu': filepath_deepseek_modu
    }
    dataset_dict = {name: Path(filepath)
    for name, filepath in dataset_dict.items()
    if filepath is not None}

    pd_datasets = {}
    mfrm_datasets = {}
    all_dataframes = {}

    for dataset_name, filepath in dataset_dict.items():
        if not filepath.is_file():
            print(f"Skipping {dataset_name}: file not found at {filepath}")
            continue
        pd_datasets[dataset_name] = pd.read_csv(filepath)
        mfrm_datasets[dataset_name] = pd_datasets[dataset_name][['annotator_id', 'comment_id'] + DIMENSIONS[:-1]].copy()
        all_dataframes[dataset_name] = data_differentiate_raters_3(
            mfrm_datasets[dataset_name], annotator_label=dataset_name)

    #each model against human for modu only
    for model in ['gpt-5-mini_modu', 'claude_modu', 'gemini_modu', 'deepseek_modu']:
        combined_name = f'llm_human_{model}'
        all_dataframes[combined_name] = pd.concat([
            all_dataframes['gold'], all_dataframes[model]], 
            ignore_index=True)
        all_dataframes[combined_name].to_csv(os.path.join(
            output_dir, f'{combined_name}_ready_{timestamp}.csv'
            ), index=False)

    #llm_modu only
    all_dataframes['llm_only_modu'] = pd.concat([
        all_dataframes[k] for k in [ 
            'gpt-5-mini_modu','claude_modu',
            'gemini_modu', 'deepseek_modu']], 
        ignore_index=True)


    all_dataframes['llm_human_modu'] = pd.concat([
            all_dataframes[k] for k in [
                'gold', 'gpt-5-mini_modu',
                'claude_modu', 'gemini_modu',
                'deepseek_modu']], 
        ignore_index=True)

    #save all
    for dataset_name, df in all_dataframes.items():
        mfrm_save = f'mfrm_{dataset_name}_ready_{timestamp}.csv'
        mfrm_csv_file = os.path.join(output_dir, mfrm_save)
        df.to_csv(mfrm_csv_file, index=False)
        print(f"✓ Saved {dataset_name} MFRM-ready data to: {mfrm_csv_file}") 

    #extra validation checks
    all_data_concat = all_dataframes['llm_human_modu']
    print(f"\nCombined dataset:")
    print(f"Total rows in concatenated filtered df: {len(all_data_concat)}")
    print(f"Unique comments: {all_data_concat['comment_id'].nunique()}")
    print(f"Unique raters: {all_data_concat['annotator_id'].nunique()}")
    # print(f"\nRater breakdown:") 
    print(f"  Rater breakdown:")
    print(all_data_concat['annotator_id'].value_counts().to_string())
    print("\nchecking for missing values")

    # Check for missing values
    missing_summary = all_data_concat[DIMENSIONS[:-1]].isnull().sum()
    if missing_summary.sum() > 0:
        print("\n[Problem]: Missing values detected:")
        print(missing_summary[missing_summary > 0])
    else:
        print("\n[GOOD] No missing values in dimension ratings")

    return pd_datasets, all_dataframes, all_data_concat


def get_severity(severity_df):
    rater_params = severity_df.copy()
    def get_llm_label(rater_id):
        rater_id    = str(rater_id)
        rater_lower = rater_id.lower()
        if rater_id.isdigit():
            return 'Human'
        elif 'gpt-5-mini' in rater_lower:
            model = 'gpt-5-mini'
        elif 'claude' in rater_lower:
            model = 'claude'
        elif 'gemini' in rater_lower:
            model = 'gemini'
        elif 'deepseek' in rater_lower:
            model = 'deepseek'
        else:
            return 'New_Model'

        #detect prompting strategy
        if 'modu' in rater_lower:
            return f'{model}_modu'
        else:
            return model

    rater_params= rater_params.reset_index()
    print(rater_params.columns)
    rater_params['rater_type'] = rater_params['Rater'].map(get_llm_label)
    
    # Reset index to make rater_id a column
    rater_params.rename(columns={'Rater':'annotator', 'Estimate': 'severity'}, inplace=True)
    rater_params_plotting = rater_params[['annotator', 'severity', 'rater_type']]
    
    return rater_params, rater_params_plotting

def plot_severity_strip(rater_params, OUTPUT_DIR=None, colors=COLORS):
    """
    Plot showing severity distribution per annotator type.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    rater_params = rater_params.copy().reset_index()
    is_human = rater_params['Rater'].astype(str).str.isdigit()
    rater_params['rater_type'] = np.where(is_human, 'Human', rater_params['Rater'])

    order = ['Human', 'gpt-5-mini_modu', 'claude_modu', 'gemini_modu', 'deepseek_modu']

    groups = [g for g in order if g in rater_params['rater_type'].unique()]

    for i, group in enumerate(groups):
        sub = rater_params[rater_params['rater_type'] == group]
        severity = sub['Estimate']
        infit = sub['Infit MS'].clip(lower=0.01)
        color  = colors.get(group, '#888888')
        n      = len(sub)
        W = 0.45
        

        if group == 'Human':
            # # Jitter x positions so points don't overlap
            jitter = np.random.uniform(-0.25, 0.25, size=n)
            x_pos  = i + jitter
            h = sub['Infit MS']
            print(f'human_infit>2 ={round((h > 2).mean(),1) * 100}%,\nhuman infit[0.5, 1.5]={round(h.between(0.5, 1.5).mean(),1) * 100}%,\nhumna_median_infit={round(h.median(),3)}')

            # Plot individual points
            ax.scatter(
                x_pos, severity,
                color=color, 
                alpha=0.4,
                s=40,
                zorder=1,
                label=f"Human_raters (n={n})"
            )
            q1, med, q3 = severity.quantile([0.25, 0.5, 0.75])
            print(f"Human severity quantiles: Q1={q1:.3f}, Median={med:.3f}, Q3={q3:.3f}")
            ax.hlines(med, i - W, i + W, color=color, lw=2.5, zorder=4)
            ax.fill_between([i - W, i + W], q1, q3, color=color, alpha=0.2, zorder=3, label='Human IQR(25-75 %)')
            ax2.scatter(x_pos, sub['Infit MS'], color=color, alpha=0.4, s=30)
            #infit
            q1_i, med_i, q3_i = sub['Infit MS'].quantile([.25, .5, .75])
            ax2.hlines(med_i, i-W, i+W, color=color, lw=2.5, zorder=4)
            ax2.fill_between([i-W, i+W], q1_i, q3_i, color=color, alpha=0.2, zorder=3,label='Human IQR(25-75 %)')
        else:
            for val, se in zip(severity, sub['SE']):
                ax.errorbar(
                    i, val, yerr=se*1.96, 
                    fmt='o', color=color, 
                    alpha=0.7, capsize=3, 
                    lw=1.8, zorder=4, label=f"{group}(SE={se:.3f})")
                ax.annotate(
                    f"{val:.2f}",
                    xy=(i + 0.05, val),
                    fontsize=12, color=color, va='center'
                )
            v = infit.iloc[0]
            ax2.scatter(i, v, color=color, alpha=0.4, s=40, zorder=1, label=f"{group}(Infit={v:.2f})")
            ax2.annotate(f"{v:.2f}", (i+0.12, sub['Infit MS'].iloc[0]),
                        color=color, fontsize=12, va='center')

     # Reference line at 0 (average severity)
    ax.axhline(0, color='black', lw=1, linestyle='--', alpha=0.6, label='Scale origin (0)')

    # # Shade the "lenient" and "harsh" regions
    y_max = rater_params['Estimate'].max() + 0.25
    y_min = rater_params['Estimate'].min() - 0.25
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, len(groups) - 0.4)
    ax.axhspan(0.25,  y_max, alpha=0.1, color='red',  label='strict(less likely to label as hate)')
    ax.axhspan(y_min, -0.25, alpha=0.1, color='blue', label='Lenient(More likely to label as hate)')

    # Formatting
    ax.set_xticks(range(len(groups)))
    ax.set_ylabel("Severity (logits)", fontsize=15)

    ax2.axhspan(2.0, 200, color='red', alpha=0.06, label='Misfit (threshold:2.0)')
    ax2.set_yscale('log')
    ax2.set_ylabel('Rater Infit MS (log)', fontsize=15)
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels(groups, fontsize=10)
    ax2.set_xlabel('Annotator', fontsize=15)

    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax2.legend(fontsize=8, loc='upper right', ncol=2)

    plt.tight_layout()

    if OUTPUT_DIR:
        plt.savefig(OUTPUT_DIR, dpi=150, bbox_inches='tight')

    plt.show()

def extract_theta(mfrm, label=""):
    mfrm.person_stats_df_global()
    theta = mfrm.person_stats_global.get('Estimate')
    theta.index.name = 'comment_id'
    if theta is None or theta.empty:
        raise ValueError(f"No theta estimates found for: {label}")
    return theta

def calibrate(filepath, label):
    """Helper to load and calibrate MFRM model."""
    print(f"  Calibrating {label}...")
    data, _ = rp.loadup_mfrm_single(filepath)
    model   = rp.MFRM(data)
    model.calibrate_global()
    return model