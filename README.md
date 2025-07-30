# Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation

> **ACL 2025 · Main Track**
> Official code for **“[Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation.](https://aclanthology.org/2025.acl-long.900/)”**

---

## 📌 Abstract
<p align="center">
<img width="304" height="453" alt="image" src="https://github.com/user-attachments/assets/59bea0cf-bb69-47ed-ac75-d2ca64a40f72" />
</p>

Personalized dialogue systems have improved markedly with the use of user-specific personas in large language models (LLMs). Yet, the **impact of persona sentiment polarity on dialogue quality** has been largely overlooked. We conduct a large-scale study with polarized user profiles and show:

* **Negative personas** push models to overuse persona attributes.
* **Positive personas** lead to selective, smoother persona integration.
* **Weak/neutral personas** generally produce lower-quality conversations.

Motivated by these findings, we propose a generation strategy that explicitly accounts for persona polarity via **turn-based generation**, **profile ordering**, and **sentiment-aware prompting**. Our work reveals how sensitive LLMs are to persona sentiment and offers guidance for building more robust, nuanced personalized dialogue systems.

---

## 🧭 Table of Contents

* [Requirements & Environment](#requirements--environment)
* [Installation](#installation)
* [Pipeline Overview](#pipeline-overview)

  1. [Build Polarized Profiles](#1-build-polarized-profiles)
  2. [Generate & Filter Dialogues](#2-generate--filter-dialogues)
  3. [Evaluate Dialogues](#3-evaluate-dialogues)
* [Citation](#citation)
* [Contact](#contact)

---

## Requirements & Environment

* **Python**: 3.10
* Recommended: CUDA-enabled GPU for faster generation/evaluation

Create and activate the conda environment:

```bash
conda create -n persona python=3.10 -y
conda activate persona
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

All scripts are under `scripts/`, and major modules live in subdirectories like `dialogue_generation/` and `evaluation/`.

### 1) Build Polarized Profiles

```bash
bash scripts/build_profiles.sh
```

> **Note:** Move all generated profiles into the `dialogue_generation/` directory before proceeding to Step 2.

---

### 2) Generate & Filter Dialogues

Generate personalized dialogues according to your **profile pairing configuration**:

* **Dual-persona joint generation**

  ```bash
  bash scripts/generate_dialogue_joint.sh
  ```

* **Turn-based generation**

  ```bash
  bash scripts/generate_dialogue_tb.sh
  ```

  You can adjust **profile ordering strategies** and **sentiment-aware prompting options** within this script.

After generation, filter dialogues using the provided criteria:

```bash
bash scripts/filter_dialogue.sh
```

> **Note:** Move the paired profiles and generated dialogues into the `evaluation/` directory before Step 3.

---

### 3) Evaluate Dialogues

Download the following folders or models from the [link](https://drive.google.com/drive/folders/1Vv0DTtifNXj3AH0H8HlJewy4vw9e0sbX?usp=drive_link) and place them under `evaluation/`:

* `nli_model`
* `QuantiDCE`
* `Paireval`

Then run the metric-specific scripts:

* **C score & Contd.**

  ```bash
  bash scripts/C2_eval.sh
  ```

* **P\_gap & Perplexity**

  ```bash
  bash scripts/P2_eval.sh
  ```

* **Q-DCE score**

  ```bash
  cd ./evaluation/QuantiDCE/
  bash script/infer.sh
  ```

* **Paireval score**

  ```bash
  cd ./evaluation/Paireval/
  python inference.py
  ```

* **LLM-based (G-Eval) score**

  ```bash
  bash scripts/G-eval.sh
  ```

The outputs will be saved for subsequent analysis.

---

## Citation

If this repository helps your research, please cite:

```bibtex
@inproceedings{jun-lee-2025-exploring,
    title = "Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation",
    author = "Jun, Yonghyun  and
      Lee, Hwanhee",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.900/",
    pages = "18384--18402",
    ISBN = "979-8-89176-251-0",
    abstract = "Personalized dialogue systems have advanced considerably with the integration of user-specific personas into large language models (LLMs). However, while LLMs can effectively generate personalized responses, the influence of persona sentiment on dialogue quality remains underexplored. In this work, we conduct a large-scale analysis of dialogues generated using a range of polarized user profiles. Our experiments reveal that dialogues involving negatively polarized users tend to overemphasize persona attributes. In contrast, positively polarized profiles yield dialogues that selectively incorporate persona information, resulting in smoother interactions. Furthermore, we find that personas with weak or neutral sentiment generally produce lower-quality dialogues. Motivated by these findings, we propose a dialogue generation approach that explicitly accounts for persona polarity by combining a turn-based generation strategy with a profile ordering mechanism and sentiment-aware prompting. Our study provides new insights into the sensitivity of LLMs to persona sentiment and offers guidance for developing more robust and nuanced personalized dialogue systems."
}

```

> Update authors and bibliographic details with the official ACL entry once available.

---

## Contact

Questions or issues? Please open a GitHub issue or reach out at **\[zgold5670@cau.ac.kr]**.

---



**Enjoy experimenting! ✨**

