# config.py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent # relative to project root

TRUST_DIR = PROJECT_ROOT/'TRUST_2026'/'data_outputs_modu'

DATA_DIR_RAW   = TRUST_DIR/ 'dspy_llm_data' / 'data' / 'raw_data'
DATA_DIR   = TRUST_DIR/ 'dspy_llm_data' / 'data' / 'patched_data'
DATA_DIR_MFRM   = TRUST_DIR/ 'dspy_llm_data' / 'data' / 'mfrm_data'

OUTPUT_modu_only = TRUST_DIR/ 'outputs_modu' / 'stats_modu_1run'
OUTPUT_modu_5runs = OUTPUT_modu_only/ 'stats_modu_5run_noHS'
DATA_modu_only = TRUST_DIR/  'data_modu' / 'mfrm_single_run'
DATA_modu_mfrm = TRUST_DIR/ 'data_modu' / 'mfrm_single_results'

#ablation
AGREE_OUTPUT_DIR_stats_nohs = OUTPUT_modu_only/ 'stats_modu_1run_2raters'

#we save all the figures here
Figures = TRUST_DIR/ 'dspy_llm_data' /'all_figures'


for _dir in [
    DATA_DIR, DATA_DIR_RAW,DATA_DIR_MFRM, 
    OUTPUT_modu_only, DATA_modu_only, DATA_modu_mfrm,
    Figures, OUTPUT_modu_5runs, AGREE_OUTPUT_DIR_stats_nohs
]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Dimensions ────────────────────────────────────────────────────────────────
DIMENSIONS = [
    'sentiment', 'respect', 'insult', 'humiliate', 'status',
    'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech'
]

ORDINAL_DIMS = DIMENSIONS[:-1]
BINARY_DIM   = 'hatespeech'

# ── Colours ───────────────────────────────────────────────────────────────────

COLORS = {
    'Human':           '#85929E',   # stone gray
    'GPT5_mono':       '#E69F00',  # deep crimson'#C0392B'
    'gpt-5-mini_mono': '#E74C3C',   # bright red
    'gpt-5-mini_modu': '#922B21',   # dark blood red
    'claude_mono':     '#1A5276',   # deep navy blue
    'claude_modu':     '#56B4E9',   # strong cobalt blue #2980B9
    'gemini_mono':     '#27AE60',   # strong emerald green
    'gemini_modu':     '#1E8449',   # deep forest green
    'deepseek_mono':   '#6C3483',   # deep violet
    'deepseek_modu':   '#9B59B6',   # strong purple
}

# ── LLM annotator names ───────────────────────────────────────────────────────
LLM_NAMES = ['GPT5_mono', 'gpt-5-mini_mono', 'gpt-5-mini_modu', 'claude_mono', 'claude_modu', 
             'gemini_mono', 'gemini_modu', 'deepseek_mono', 'deepseek_modu']


ANNOTATOR_ID_MAP = {
    'GPT5_mono'              : 'GPT5_mono',
    'gpt-5-mini_mono'        : 'gpt-5-mini_mono',
    'gpt-5-mini_modu'        : 'gpt-5-mini_modu',
    'claude-haiku-4.5_mono'  : 'claude_mono',
    'claude-haiku-4.5_modu'  : 'claude_modu',
    'gemini-3.6-flash_mono'  : 'gemini_mono',
    'gemini-3.6-flash_modu'  : 'gemini_modu',
    'deepseek-v4-flash_mono' : 'deepseek_mono',
    'deepseek-v4-flash_modu' : 'deepseek_modu',
}
