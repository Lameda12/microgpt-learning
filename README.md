# 🧠 MicroGPT Learning Journey

Welcome! This repository documents an interactive deep-dive into the core architecture of Large Language Models (LLMs), built during my 2nd year as a Computer Science student at Dalhousie University. 

The goal of this project was to demystify AI: moving away from "black-box" APIs and actually understanding the math, the forward pass, and the calculus of backpropagation that powers state-of-the-art models like ChatGPT.

> **💡 Core Philosophy:** "What I cannot create, I do not understand." - *Richard Feynman*

---

## 📜 Acknowledgements & Citations
This repository is heavily inspired by and directly cites the educational work of **Andrej Karpathy** (former Director of AI at Tesla, founding member of OpenAI). 

The baseline pure-Python architecture (`microgpt.py`) is adapted from his incredible Gist:
🔗 **[Andrej Karpathy's `microgpt.py` Gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95#file-microgpt-py)**

---

## 🚀 What's in this Repository?

We built and experimented with two distinct implementations of the GPT (Generative Pre-trained Transformer) architecture to understand the trade-offs between mathematical transparency and commercial scalability.

### 1. The Pure Python Version (`microgpt.py` / `shakespeare_gpt.py`)
This script implements a GPT from the absolute ground up with **zero dependencies**—no PyTorch, no TensorFlow, no NumPy. 
* **The Autograd Engine:** We used a custom `Value` class that wraps individual floating-point scalars and dynamically builds a computation graph to calculate gradients via the chain rule.
* **The Lesson:** While this is a brilliant way to learn how the math works, running a Transformer on scalar Python lists is incredibly slow. Our tiny 4,000-parameter model took minutes to process just a few batches of Shakespeare text!

### 2. The Commercial PyTorch Version (`shakespeare_pytorch.py`)
To see how real AI companies train models, we rewrote the architecture using **PyTorch**, replacing the custom autograd engine with highly optimized C++/CUDA tensors.
* **The Scale:** Because PyTorch handles vector-math instantly, we scaled the model up from 4,192 parameters to **210,000 parameters**.
* **The Lesson:** The real speed upgrade in AI engineering comes from Tensor operations. Despite being 50x larger, the PyTorch model trained in a fraction of the time and quickly learned to hallucinate realistic-looking Shakespearean dialogue (including proper capitalization, syntax, and custom character names!).

---

## 🤯 The "Aha!" Moment: How GPT-2 and GPT-3 Were Initiated
The most valuable lesson from this session was uncovering the "secret" of scale. 

What is running in `shakespeare_pytorch.py` is **the exact same foundational architecture as GPT-2 and GPT-3**. The only difference between our toddler-level Shakespeare bot and ChatGPT is simply scale and data. 

To turn this repository into GPT-3, you would theoretically just need to:
1. **Scale the Tokenizer:** Swap our character-level tokenization for Byte-Pair Encoding (BPE) to process whole syllables/words.
2. **Scale the Data:** Instead of a 1MB file of Shakespeare, train it on a 10TB scrape of the entire internet.
3. **Scale the "Brain" Dimensions:** Change `n_layer` from 4 to 96, and `n_embd` from 64 to 12288. You instantly have a 175-Billion parameter brain! (You just need a few million dollars in GPU compute to calculate the gradients 😅).
4. **Fine-Tuning (RLHF):** Teach the raw statistical engine to format its output as a helpful assistant rather than just blindly completing internet documents.

---

## 🤝 Open for Contributions!
This repository is an open learning sandbox. If you are a fellow student, hobbyist, or AI enthusiast, feel free to clone this repo and run the training loops on your own laptop!

**Ideas for Pull Requests:**
- [ ] Add a `generate.py` script that lets users type a starting prompt (e.g., `"ROMEO: "`) and have the model auto-complete the rest.
- [ ] Implement a function to save the PyTorch `state_dict` (brain) to a `.pt` file so we don't have to retrain every time.
- [ ] Try training the PyTorch model on a new dataset (like a list of startup names, or math equations) to see what it learns!

Fork it, break it, learn from it. Contributions and discussions are highly welcome!
