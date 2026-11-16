# Auditing LLM Judge for Hate Speech Annotation with Psychometric Fit Statistics and Agreement Metrics.
The code runs four LLM judges on 250 comments from the Measuring Hate Speech (MHS) corpus, calibrates them with a many-facet Rasch model (MFRM), and compares what agreement metrics and Infit/Outfit fit statistics each flag.

## Dataset Used 
We use the Measuring Hate Speech corpus by Kennedy et al. (2020). See [MHS Corpus on HugginFace](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech).
```
from datasets import load_dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
```

## Models used
    | Model               | Provider   | Type   | Origin      |
    |---------------------|------------|--------|-------------|
    | GPT-5-mini          | OpenAI     | Closed | Western     |
    | Claude Haiku 4.5    | Anthropic  | Closed | Western     |
    | Gemini 3.6 Flash    | Google     | Closed | Western     |
    | DeepSeek-V4         | DeepSeek   | Open   | Non-Western |

## Setup

### Clone the repository

```
git clone https://github.com/yourusername/TRUST_2026.git

```

### RaschPy

```
This repository includes a local fork of [RaschPy](https://github.com/MarkElliott999/RaschPy). This is version `0.1.0`.
See `RaschPy_aies/` for the modified source.
```

### Read and activate environment

```
conda create -n hate-speech-llm python=3.11
conda activate hate-speech-llm
pip install -r requirements.txt
```

### configure API Keys
```
cp .env.example .env
```

edit the `.env` with your personal API keys

```
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
DEEPSEEK_API_KEY=your-key-here
TOGETHER_API_KEY=your-key-here
CMU_OPENAI_API_KEY=your-key-here
```
### Path setup

Edit `config.py` and set the `BASE_DIR` to your local file directory path
```
BASE_DIR = Path("/your/local/path/to/project")
```
### Slur Database

The slur database (`rsdb_slurs.json`) is not included in this 
repository. To generate it run:

```bash
python Annotation/scrape_rsdb.py
```

This scrapes the Racial Slur Database and saves it locally.

### Run annotation

Update the model confidureagtion in `lm`
Run all annotation for that model using
```
python Annotation/experiment.py
```

### Run MFRM anlaysis (Plots, figures and stats)

Import all function from the   `MFRM_analysis.py`, Implement them in the `MFRM_1run_TRUST.ipynb`


## Repository Layout and Set up

    Main_folder
    │
    TRUST_2026/    
    ├── README.md
    ├── config.py                   # Paths, dimensions, colors, model maps
    ├── requirment.txt              # Single entry point
    ├── .env                        # API key used and setup
    ├── .gitignore
    │
    ├── Annotation/                 # LLM annotation (needs API keys)
    │   ├── experiment.py           # Main annotation runner (5 runs per comment)
    │   ├── signatures.py           # DSPy signature definitions
    │   ├── dimension_rating.py     # Modular raters
    │   ├── majority_vote.py        # Vote distribution function
    │   ├── scrape_rsbd.py          # building the slur tagging database
    │   ├── patching.py             # Reannotate incomplete run patching
    │   └── utils.py                # Helper functions
    |
    ├── mfrm_analysis.py           # MFRM calibration pipeline
    ├── RaschPy_aies/              # local fork of RaschPy (install with setup.py)
    │
    ├── notebooks/
    │   ├── AGREE_ANALYSIS_TRUST.ipynb   # agreement metrics, gaps, Figure/Tables for RQ2
    │   └── MFRM_1RUN_TRUST.ipynb        # MFRM fits for the single-run analysis
    │
    └── data_output_modu/
        ├── dspy_llm_data/         # LLM ratings
        │   ├── raw_data/          # original LLM output (before patching)
        │   ├── patched_data/      # final ratings after re-annotating incomplete runs
        │   └── mfrm_data/         # data prepared for MFRM calibration
        ├── data_modu/
        │   ├── mfrm_single_run/   # single-run LLM data
        │   └── mfrm_single_result/# joint and pooled calibrations for the single runs
        └── output_modu/
            ├── figures/           # figures reported in the paper
            └── stats_modu_1run/   # tables and results reported in the paper

## Setup

```bash
pip install -r requirements.txt
pip install -e RaschPy_aies        # local RaschPy fork
cp .env.example .env               # then add your own API keys
```

Required keys: `[OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY]'.

Python3 version: `3.11.14`. 

DSPy version: `3.1.0` with `cache=False` ( extra precaution; `litellm.cache = None`)so that each of the five runs made fresh calls. 

Comments were shuffled with `seed = 42`

## Data notes

- **Human ratings and comments** come from the public [MHS Corpus on HugginFace](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
- **Five runs per comment.** Ratings for the five runs were consolidated per comment and dimension into `vote_distributions` (counts per rating value). 
- **Calibrations.** "All-rater pooled" fits all humans and LLMs on one scale; while the "Joint" fits all the raters (human + all LLM) on one scale.


## Citation
If you use this work please cite:

## Acknowledgements

- Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application. [Kennedy et al. 2022](https://arxiv.org/abs/2009.10277)
- DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. [Khattab et al. 2023](https://arxiv.org/abs/2310.03714)
- [RaschPy](https://github.com/MarkElliott999/RaschPy) for the open-source Python MFRM implementation