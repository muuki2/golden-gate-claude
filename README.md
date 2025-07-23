# Golden Gate Claude: Interpreting and Steering LLMs via Feature Activation

This repository contains the notebook **Golden_Gate_Claude.ipynb**, which demonstrates how to identify, extract, and steer a "feature" (concept) within a large language model (LLM) — specifically, the concept of the Golden Gate Bridge — using techniques inspired by the Anthropic research paper ["Golden Gate Claude"](https://www.anthropic.com/news/golden-gate-claude).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muuki2/golden-gate-claude/blob/main/Golden_Gate_Claude.ipynb)
## Background

Anthropic's "Golden Gate Claude" project explores the interpretability of LLMs by identifying internal neuron activations corresponding to specific concepts (features), such as the Golden Gate Bridge. By manipulating these activations, researchers can directly influence the model's behavior in a targeted, interpretable way — for example, making the model mention the Golden Gate Bridge more often, regardless of the prompt.

This notebook adapts these ideas to an open-source LLM (Qwen/Qwen3-1.7B), showing how to:

- Extract a "feature vector" for the Golden Gate Bridge from the model's activations.
- Steer the model's responses by injecting this vector at a specific layer.
- Observe the resulting change in model behavior.

## Notebook Overview

### 1. **Setup and Dependencies**
- Installs and imports required libraries: `transformer_lens`, `einops`, `circuitsvis`, etc.
- Downloads the ARENA_3.0 exercises for interpretability utilities.

### 2. **Model Loading**
- Loads the Qwen3-1.7B model using `transformer_lens`.
- Ensures GPU usage for efficient computation.

### 3. **Prompt Formatting**
- Implements a function to format prompts for the Qwen3 chat model.

### 4. **Golden Gate Bridge Feature Extraction**
- Loads a set of Golden Gate Bridge-related conversations.
- Tokenizes and runs them through the model to collect activations.
- Identifies the "feature vector" corresponding to the Golden Gate Bridge by averaging the residual stream activations for relevant tokens.

### 5. **Model Steering**
- Defines a steering hook that injects the Golden Gate Bridge feature vector at a chosen layer.
- Demonstrates how this changes the model's output to focus on the Golden Gate Bridge, even for unrelated prompts.

### 6. **Experimentation**
- Provides code to run the steered model and observe the effect.

## Example

After running the notebook, you can ask the model a generic question like:

> "What is your favourite place to go?"

With the steering vector applied, the model will respond with something like:

> "The Golden Gate Bridge in San Francisco is one of the most famous landmarks..."

## How to Use

1. **Open the notebook**: `Golden_Gate_Claude.ipynb` in Jupyter or Google Colab.
2. **Run all cells**: The notebook will install dependencies, download data, and set up the model.
3. **Experiment**: Try different prompts and adjust the steering scale to see how the model's behavior changes.

## Requirements

- Python 3.8+
- GPU (recommended for speed)
- Packages: `transformer_lens`, `einops`, `circuitsvis`, `jaxtyping`, `torch`, `huggingface_hub`, etc.

All dependencies are installed automatically in the first cell.

## Reference

- **Anthropic Research Paper**: ["Golden Gate Claude"](https://www.anthropic.com/news/golden-gate-claude)
- **ARENA 3.0**: [https://github.com/callummcdougall/ARENA_3.0](https://github.com/callummcdougall/ARENA_3.0)
- **Qwen3-1.7B Model**: [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-1.7B)

## License

This notebook is for educational and research purposes only. Please refer to the respective licenses of the ARENA 3.0 repository and the Qwen3-1.7B model.

---

**Inspired by Anthropic's work on LLM interpretability and feature steering.**  
For more details, see the [Golden Gate Claude announcement and research summary](https://www.anthropic.com/news/golden-gate-claude).
