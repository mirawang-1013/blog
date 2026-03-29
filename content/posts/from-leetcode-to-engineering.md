---
title: "从 LeetCode 到工程代码：我读 LangChain 源码并从零搭了一个 RAG 系统"
date: "2026-03-29"
summary: "刷了几百道 LeetCode，却发现看不懂 GitHub 上的项目代码？这篇文章记录了我如何通过阅读 LangChain 源码，学会读代码的方法论，并从零搭建一个基于自己 Obsidian 笔记的 RAG 问答系统。"
tags: ["AI", "RAG", "LangChain", "engineering"]
---

> 刷了几百道 LeetCode，却发现看不懂 GitHub 上的项目代码？这篇文章记录了我如何通过阅读 LangChain 源码，学会读代码的方法论，并从零搭建一个基于自己 Obsidian 笔记的 RAG 问答系统。

## LeetCode 和真实项目的鸿沟

| | LeetCode | 真实项目 |
|--|---------|---------|
| 规模 | 一个函数 | 几百个文件互相调用 |
| 目标 | 解决一个算法问题 | 让代码可扩展、可维护、可复用 |
| 思维 | 「怎么算出答案」 | 「怎么组织代码让别人能看懂、能改」 |
| 核心能力 | 算法 + 数据结构 | 抽象 + 接口设计 + 模块拆分 |

LeetCode 练的是**计算思维**，真实项目练的是**工程思维**。两个都需要，但中间有个鸿沟。我决定通过阅读 LangChain 源码来跨过它。

---

## Part 1: 怎么读一个大型开源项目

### 不要从头读到尾，要像侦探一样追线索

拿到 LangChain 的仓库后，我的第一反应是懵的 — 几百个文件从哪看起？后来我学到了一套方法论：

**第一步：看目录结构，搞清楚模块划分**

```
libs/
├── core/          # 基础抽象层 — 一切的根基
├── langchain/     # 主包 — chains, agents, retrieval
├── partners/      # 第三方集成（OpenAI, Anthropic 等）
├── text-splitters/ # 文档切分
└── standard-tests/ # 共享测试
```

**第二步：找到核心抽象**

怎么知道哪个是核心？几个技巧：
- 读 README，好项目会告诉你核心概念
- 看 `__init__.py` 导出了什么 — 最重要的东西一定在顶层
- 用 `grep` 看什么被 import 最多 — 被依赖最多的就是核心

对 LangChain 来说，核心是 `Runnable` — 一切组件都实现这个接口。

**第三步：选一条调用链，追到底**

我选了 `ChatModel.invoke()` 这条线。用 `grep` 定位，逐层深入：

```bash
# 找方法定义的位置
grep -n "def invoke" libs/core/langchain_core/language_models/chat_models.py

# 看具体实现
sed -n '441,470p' libs/core/langchain_core/language_models/chat_models.py
```

### 追踪 invoke 的完整调用链

源码里 `invoke` 的实现是这样的（简化版）：

```python
def invoke(self, input, config, ...) -> AIMessage:
    config = ensure_config(config)
    return self.generate_prompt(
        [self._convert_input(input)], ...
    ).generations[0][0].message
```

拆解调用链：

```
invoke(input)
  → _convert_input(input)    # 把各种输入格式统一成 PromptValue
  → generate_prompt(...)      # 调用模型 API
  → .generations[0][0]        # 取第一个结果
  → .message                  # 提取 AIMessage
```

### 读嵌套代码的技巧：从里往外剥

原始代码里有很多 `cast()` 嵌套，看起来很吓人。技巧是：**忽略 cast（只是类型标注），从最内层往外读。**

### _convert_input：宽进严出的设计模式

```python
def _convert_input(self, model_input):
    if isinstance(model_input, PromptValue):
        return model_input
    if isinstance(model_input, str):
        return StringPromptValue(text=model_input)
    if isinstance(model_input, Sequence):
        return ChatPromptValue(messages=convert_to_messages(model_input))
    raise ValueError(...)
```

不管你传 string、message list、还是 PromptValue，它都能处理。入口接受多种格式，内部统一处理 — 这叫**适配器模式**。

---

## Part 2: LangChain 的核心设计思想

### 为什么用 LangChain 而不是直接调 OpenAI SDK？

一个字：**可替换性**。

假设要从 OpenAI 换成 Claude：

**直接用 SDK** — API 不一样、参数不一样、返回格式不一样，每个调用都要改。

**用 LangChain** — 只改一行：

```python
# 之前
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# 只改这里
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# 下面所有代码不用动
answer = llm.invoke(prompt)     # 接口一样
answer.content                   # 返回格式一样
```

因为所有模型都实现了同一个 `Runnable` 接口。这就是 `BaseChatModel` 的价值 — 上层代码不需要知道底层是哪个模型。

**核心启发：反复调用的功能应该写成组件，统一输入和输出，这样代码更干净、更容易替换。**

---

## Part 3: 从零搭一个 RAG 系统

读完源码后，我决定动手写一个实际项目：**一个基于 Obsidian 笔记的智能问答系统**。

### 项目结构

```
obsidian_rag/
├── loader.py      # 读取和切分笔记
├── retriever.py   # 向量化 + 相似度搜索 + 缓存
└── main.py        # 入口：串联整个流程
```

三个文件，各自职责清晰，互不耦合。

### loader.py — 加载数据

职责：遍历所有 markdown 文件，按段落切分，加上来源标记。

```python
import os

def load_notes(notes_dir: str) -> list[str]:
    notes = []
    for file in os.listdir(notes_dir):
        if file.endswith(".md"):
            with open(os.path.join(notes_dir, file), "r", encoding="utf-8") as f:
                content = f.read()
                paragraphs = content.split("\n\n")
                for paragraph in paragraphs:
                    if len(paragraph.strip()) < 20:
                        continue
                    notes.append(f"[{file}] {paragraph.strip()}")
    return notes
```

关键设计：每个段落带上文件名前缀 `[filename.md]`，这样检索出来的结果能追溯来源。

### retriever.py — 向量检索 + 增量缓存

这是整个项目最核心的部分。

**余弦相似度：衡量两个向量"方向有多像"**

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

值在 -1 到 1 之间，越大越相似。比如 "ResNet 残差网络" 和 "深度学习残差连接" 的向量相似度会很高。

**增量缓存：用 set 差集实现**

每次 embedding 1000+ 条笔记要调 API，很慢也花钱。解决方案是把向量缓存到本地文件，下次只处理新增的部分：

```python
import pickle
import os
from tqdm import tqdm

def build_index(notes, embeddings, cache_path="index.pkl"):
    index = load_index(cache_path)
    if index is None:
        index = {}

    cached_keys = set(index.keys())
    current_keys = set(notes)
    new_keys = current_keys - cached_keys       # 新增的段落
    removed_keys = cached_keys - current_keys   # 已删除的段落

    for key in tqdm(new_keys, desc="Embedding 新增段落"):
        index[key] = embeddings.embed_query(key)
    for key in removed_keys:
        index.pop(key)

    save_index(index, cache_path)
    return index
```

核心思路：
- `current_keys - cached_keys` = 新增的，需要做 embedding
- `cached_keys - current_keys` = 删掉的，从缓存移除
- 其余的 = 没变的，直接跳过

第一次跑：全部是 "新增"，全量 embedding。第二次跑：差集为空，秒开。

**检索函数**

```python
def retrieve(index, query, embeddings, top_k=3):
    query_embedding = embeddings.embed_query(query)
    similarities = sorted(
        index.items(),
        key=lambda x: cosine_similarity(x[1], query_embedding),
        reverse=True
    )
    return [item[0] for item in similarities[:top_k]]
```

### main.py — 串联整个流程

```python
from dotenv import load_dotenv
from retriever import build_index, retrieve
from loader import load_notes
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def get_response(query, results, llm):
    context = "\n\n".join(results)
    prompt = f"""根据以下笔记内容回答问题。如果笔记里没有相关信息，就说不知道。

笔记内容：
{context}

问题：{query}"""
    response = llm.invoke(prompt)
    print(response.content)

def main():
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-mini")

    print("加载笔记中...")
    notes = load_notes("/Users/hipvan/Downloads/Obsidian_notes")
    index = build_index(notes, embeddings)
    print(f"索引建好了，共 {len(notes)} 条段落")

    query = input("Hi, what do you want to know? (输入 exit 退出): ")
    while query.lower() != "exit":
        results = retrieve(index, query, embeddings)
        get_response(query, results, llm)
        query = input("\n还想问什么? (输入 exit 退出): ")

if __name__ == "__main__":
    main()
```

`get_response` 里的 prompt 就是 RAG 的核心：**把检索到的笔记内容塞进 prompt，让 LLM 基于你的数据回答**。

### 运行效果

```
Hi, what do you want to know?: what is CNN?

A CNN, or Convolutional Neural Network, is a type of deep learning model
primarily used for processing structured grid data, such as images...
关键组件包括：Conv2D, ReLU, Softmax, Max Pool2D, Global Average Pooling, Dropout...
```

它真的从我的论文笔记里提取了相关内容来回答！

---

## Part 4: 从源码阅读中学到的设计模式

| 在 LangChain 里看到的 | 可以用在自己项目里的思路 |
|---------|----------------------|
| `Runnable` 统一接口 | 定义基类，所有组件实现同一个方法，就能自由组合 |
| `_convert_input` 适配多种输入 | 函数入口做类型转换，内部只处理一种格式 |
| `core` 和 `partners` 分离 | 核心逻辑和第三方集成分开，方便扩展 |
| `BaseCache` 抽象类 | 定义接口，具体实现（内存 / 文件 / 数据库）随时替换 |

---

## 总结：跨越鸿沟的方法

LeetCode 教你写**正确的代码**，工程能力教你写**别人能用的代码**。

跨越的方法就三步：

1. **读源码学方法论** — 不是从头读，而是 grep 定位 → 追调用链 → 理解设计模式
2. **写小项目练手** — 每次刻意练一个模式（模块拆分、接口抽象、缓存设计）
3. **从小到大** — 100 行工具 → 500 行项目 → 开源贡献

这个 RAG 项目只有三个文件，但覆盖了 AI Engineer 面试最常考的知识点：文档加载、向量化、相似度检索、Prompt 工程、增量缓存。而且因为是自己一行行写的，理解深度和照着教程抄完全不一样。

---

*项目地址：[obsidian_rag](https://github.com/mirawang-1013/obsidian_rag)*

*如果你也在从 LeetCode 向工程代码过渡，欢迎交流。*
