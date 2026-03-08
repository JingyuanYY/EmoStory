# EmoStory: Emotion-Aware Story Generation

**Project page:** https://github.com/Chanshang/EmoStory.git

**Paper (arXiv):** https://arxiv.org/abs/111

## Abstract
Story generation aims to produce image sequences that depict coherent narratives while maintaining subject consistency across frames. Although existing methods have excelled in producing coherent and expressive stories, they remain largely emotion-neutral, focusing on what subject appears in a story while overlooking how emotions shape narrative interpretation and visual presentation. As stories are intended to engage audiences emotionally, we introduce emotion-aware story generation, a new task that aims to generate subject-consistent visual stories with explicit emotional directions. This task is challenging due to the abstract nature of emotions, which must be grounded in concrete visual elements and consistently expressed across a narrative through visual composition. 

To address these challenges, we propose **EmoStory**, a two-stage framework that integrates agent-based story planning and region-aware story generation. The planning stage transforms target emotions into coherent story prompts with emotion agent and writer agent, while the generation stage preserves subject consistency and injects emotion-related elements through region-aware composition.

<p>
    <img src="docs/teaser_page-0001.jpg" width="800px"/>  
    <br/>
    A training-free method for story generation that stimulate the designated emotions.
</p>


## Installation

### Clone the repository

```bash
# 1) Clone the repository
git clone https://github.com/Chanshang/EmoStory.git
cd EmoStory

# 2) Create and activate env
conda create -n emostory python=3.12
conda activate emostory

# 3) Install the compatible torch version
https://pytorch.org/get-started/previous-versions/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url ......

# 4) Install dependencies
pip install -r requirements.txt
```

> Tip: It is recommended to manually install the larger library by referring to the `req without dependence.txt` file, and follow the debug reminder for installation during runtime.

---

## Start

### 1. Build a story script based on the emotion factor tree and LLM

- Set LLM API interface in file `ask_gpt/Coordinated_Agent.py`.

```bash
client = OpenAI(
    base_url='https://api.openai.com/v1',
    api_key='your_api_key'
)
```

- Then run it.
```bash
python ask_gpt/Coordinated_Agent.py
```

- Generated story script for eight emotion are written to `--results/script`.  

### 2. Running an emotional enhancement story generation model


### Concrete example
- EmoStory has a high memory requirement, but can run in parallel on two 24GB GPUs. To specify GPU usage, modify the `GPU-PAIRS` in the `bash.run.sh` file. Modify `STORY_DIR` to select the script position in `results`.

```bash
bash bash_run.sh
```

- Generated story images for eight emotion are written to `--results/script`.  

---
