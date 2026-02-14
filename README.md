# 🧠 MicroGPT — The Most Atomic GPT Implementation

> 🚧 **This project is a work in progress.** Features, optimizations, and additional language ports may be added over time.

A multi-language port of Andrej Karpathy's beautifully minimal GPT trainer. One file per language. Zero (or near-zero) external dependencies. The complete algorithm — everything else is just efficiency.

---

## 📖 Overview

MicroGPT implements a **complete GPT-2 style language model** from scratch — including autograd, training, and inference — in a single file per language. It trains on a character-level dataset (names by default) and learns to generate novel, hallucinated names.

The project is a faithful port of the original Python implementation across **five languages**:

| Language | File | Dependencies | Compile / Run |
|:---------|:-----|:-------------|:--------------|
| 🐍 Python | `microgpt.py` | None (stdlib only) | `python microgpt.py` |
| ⚙️ C++ | `microgpt.cpp` | STL only | `g++ -std=c++17 -O2 -o microgpt microgpt.cpp && ./microgpt` |
| 🦀 Rust | `microgpt.rs` | `std` only | `rustc -O microgpt.rs -o microgpt && ./microgpt` |
| 🍎 Swift | `microgpt.swift` | `Foundation` only | `swiftc -O microgpt.swift -o microgpt && ./microgpt` |
| 🎯 Dart | `microgpt.dart` | `dart:io` + `dart:math` | `dart run microgpt.dart` |

> **Key principle**: Every port preserves the exact same structure, variable naming conventions, and algorithmic flow as the original Python. Reading any version is reading the same algorithm.

---

## 🏗️ Architecture

The algorithm is structured into six cleanly separated sections, identical across all languages:

### 1. Data Loading & Tokenization

```
input.txt → list of documents → character-level tokenizer
```

- Reads `input.txt` (a newline-separated list of names)
- Extracts unique characters as tokens (IDs `0..n-1`)
- Adds a special **BOS** (Beginning of Sequence) token as the final ID
- Shuffles the dataset with a fixed seed (`42`) for reproducibility

### 2. Autograd Engine (`Value`)

A scalar-valued autograd system that tracks computation graphs and computes gradients via reverse-mode differentiation (backpropagation).

**Supported operations:**
| Operation | Python | C++ | Rust | Swift | Dart |
|:----------|:-------|:----|:-----|:------|:-----|
| Addition | `a + b` | `a + b` | `a.add(&b)` | `a + b` | `a + b` |
| Multiplication | `a * b` | `a * b` | `a.mul(&b)` | `a * b` | `a * b` |
| Power | `a ** n` | `vpow(a, n)` | `a.vpow(n)` | `a.vpow(n)` | `a.vpow(n)` |
| Log | `a.log()` | `vlog(a)` | `a.vlog()` | `a.vlog()` | `a.vlog()` |
| Exp | `a.exp()` | `vexp(a)` | `a.vexp()` | `a.vexp()` | `a.vexp()` |
| ReLU | `a.relu()` | `vrelu(a)` | `a.relu()` | `a.relu()` | `a.relu()` |
| Negation | `-a` | `-a` | `a.neg()` | `-a` | `-a` |
| Subtraction | `a - b` | `a - b` | `a.sub(&b)` | `a - b` | `a - b` |
| Division | `a / b` | `a / b` | `a.div(&b)` | `a / b` | `a / b` |

**Backward pass** uses topological sorting (DFS) to propagate gradients through the computation graph — the chain rule, applied recursively.

### 3. Parameter Initialization

Weights are initialized as 2D matrices of `Value` nodes, drawn from a Gaussian distribution (`mean=0, std=0.08`):

```
state_dict:
  wte        → [vocab_size × n_embd]      Token embeddings
  wpe        → [block_size × n_embd]      Position embeddings
  lm_head    → [vocab_size × n_embd]      Output projection

  Per layer:
    attn_wq  → [n_embd × n_embd]          Query projection
    attn_wk  → [n_embd × n_embd]          Key projection
    attn_wv  → [n_embd × n_embd]          Value projection
    attn_wo  → [n_embd × n_embd]          Output projection
    mlp_fc1  → [4·n_embd × n_embd]        MLP up-projection
    mlp_fc2  → [n_embd × 4·n_embd]        MLP down-projection
```

**Default hyperparameters:**

| Parameter | Value | Description |
|:----------|:------|:------------|
| `n_embd` | 16 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 1 | Number of transformer layers |
| `block_size` | 16 | Maximum sequence length |
| `head_dim` | 4 | Dimension per head (`n_embd / n_head`) |

### 4. Model Architecture

Follows **GPT-2** with minor simplifications:

```
                    ┌──────────────────────────────────┐
                    │         Token + Position         │
                    │           Embedding              │
                    └──────────┬───────────────────────┘
                               │
                          ┌────▼────┐
                          │ RMSNorm │
                          └────┬────┘
                               │
              ┌────────────────▼────────────────┐
              │     Multi-Head Self-Attention    │
              │  (Q, K, V projections + output)  │
              │  with causal KV-cache            │
              └────────────────┬────────────────┘
                               │ + residual
                          ┌────▼────┐
                          │ RMSNorm │
                          └────┬────┘
                               │
              ┌────────────────▼────────────────┐
              │          MLP Block               │
              │  FC1 → ReLU → FC2                │
              └────────────────┬────────────────┘
                               │ + residual
                          ┌────▼────┐
                          │ lm_head │
                          └────┬────┘
                               │
                          [ logits ]
```

**Differences from standard GPT-2:**
- **LayerNorm → RMSNorm** (no learned scale/bias)
- **GeLU → ReLU** (simpler activation)
- **No biases** anywhere in the model

### 5. Training Loop

```
For each step (1..1000):
  1. Pick a document, tokenize it with BOS delimiters
  2. Forward each token through the model (autoregressive)
  3. Compute cross-entropy loss against next-token targets
  4. Backward pass (compute all gradients)
  5. Adam optimizer update with:
     - Linear learning rate decay
     - Bias-corrected first moment (β₁ = 0.85)
     - Bias-corrected second moment (β₂ = 0.99)
```

**Optimizer hyperparameters:**

| Parameter | Value |
|:----------|:------|
| Learning rate | 0.01 (linearly decayed to 0) |
| β₁ | 0.85 |
| β₂ | 0.99 |
| ε | 1e-8 |
| Steps | 1000 |

### 6. Inference

After training, the model generates 20 new samples:

```
For each sample:
  1. Start with BOS token
  2. Feed through model to get next-token probabilities
  3. Apply temperature scaling (0.5 — slightly conservative)
  4. Sample from the probability distribution
  5. Repeat until BOS is generated or block_size is reached
```

---

## ⚡ Language-Specific Design Decisions

### C++ (`microgpt.cpp`)

- **Value ownership**: `std::shared_ptr<Value>` enables shared graph node references
- **Operator overloads**: Global `operator+`, `operator*`, etc., for natural mathematical syntax
- **RNG**: `std::mt19937` (Mersenne Twister) + `std::normal_distribution` + `std::discrete_distribution`
- **Requires**: C++17 (`-std=c++17`) for structured bindings

### Rust (`microgpt.rs`)

- **Value ownership**: `Rc<RefCell<ValueInner>>` — Rust's interior mutability pattern for shared mutable nodes
- **No external crates**: Everything is `std`-only, including a hand-rolled **xoshiro256\*\*** PRNG
- **Gaussian sampling**: Box-Muller transform (no external dependency needed)
- **Weighted sampling**: Hand-rolled cumulative distribution function

### Swift (`microgpt.swift`)

- **Value as class**: Reference semantics (like Python) via Swift's `class` keyword
- **Operator overloads**: Global `+`, `*`, `-`, `/` operators matching Python's syntax
- **Identity tracking**: `ObjectIdentifier` for the visited set in topological sort
- **RNG**: Hand-rolled **xoshiro256\*\*** + Box-Muller for Gaussian

### Dart (`microgpt.dart`)

- **Value class**: Dart classes are reference types by default (perfect fit)
- **Operator overloads**: `operator +`, `operator *`, `operator -`, `operator /`
- **Named methods**: `vpow()`, `vlog()`, `vexp()`, `relu()` for operations Dart can't overload
- **RNG**: Hand-rolled **xoshiro256\*\*** + Box-Muller for Gaussian
- **String handling**: Character-by-character via index access (`string[i]`)

---

## 📂 Project Structure

```
microgpt/
├── input.txt           # Training dataset (32K+ names)
├── microgpt.py         # Original Python implementation
├── microgpt.cpp        # C++17 port
├── microgpt.rs         # Rust port
├── microgpt.swift      # Swift port
├── microgpt.dart       # Dart port
└── README.md           # This file
```

---

## 🚀 Quick Start

### Prerequisites

- An `input.txt` file in the working directory (the Python version auto-downloads it; other versions expect it to exist)

**To auto-download `input.txt`:**
```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
```

### Run any version

```bash
# Python (original)
python microgpt.py

# C++
g++ -std=c++17 -O2 -o microgpt_cpp microgpt.cpp && ./microgpt_cpp

# Rust
rustc -O microgpt.rs -o microgpt_rs && ./microgpt_rs

# Swift
swiftc -O microgpt.swift -o microgpt_swift && ./microgpt_swift

# Dart
dart run microgpt.dart
```

### Expected Output

```
num docs: 32033
vocab size: 27
num params: 7185
step    1 / 1000 | loss 3.2958
step    2 / 1000 | loss 3.2730
...
step 1000 / 1000 | loss 2.1042

--- inference (new, hallucinated names) ---
sample  1: Maliya
sample  2: Kori
sample  3: Aden
...
```

> **Note**: Exact output values may vary slightly between languages due to differences in floating-point arithmetic and RNG implementations, but the overall behavior and convergence should be consistent.

---

## 📊 Dependency Comparison

| Language | External Deps | Total Imports |
|:---------|:-------------|:--------------|
| Python | 0 | `os`, `math`, `random` |
| C++ | 0 | `<iostream>`, `<fstream>`, `<vector>`, `<cmath>`, `<random>`, `<algorithm>`, `<set>`, `<map>`, `<string>`, `<sstream>`, `<memory>`, `<functional>`, `<numeric>` |
| Rust | 0 | `std::cell`, `std::collections`, `std::fs`, `std::rc` |
| Swift | 0 | `Foundation` |
| Dart | 0 | `dart:io`, `dart:math` |

Every implementation achieves **zero external dependencies** — only standard library / language built-ins are used.

---

## 🤔 Why This Exists

This project demonstrates that:

1. **A GPT can be implemented in ~200-300 lines** in any modern language
2. **Autograd doesn't require a framework** — it's just the chain rule + a topological sort
3. **The algorithm is language-agnostic** — the same structure maps naturally to C++, Rust, Swift, and Dart
4. **Dependencies are optional** — even RNG and Gaussian sampling can be done from scratch

It's a powerful educational tool for understanding transformers at the most fundamental level.

---

## 🙏 Credits & Acknowledgments

> **Massive shoutout to [Andrej Karpathy](https://github.com/karpathy)** for the original Python implementation. His work on making deep learning accessible — from [micrograd](https://github.com/karpathy/micrograd) to [nanoGPT](https://github.com/karpathy/nanoGPT) to this beautifully minimal `microgpt.py` — has been an incredible gift to the ML community. This multi-language port is a tribute to the elegance of his original algorithm. 🎉

---

<p align="center">
  <i>"This file is the complete algorithm. Everything else is just efficiency."</i>
  <br>— Andrej Karpathy
</p>
