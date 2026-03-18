# 双进程 AGI 架构：潜意识与表意识的共生系统
# Dual-Process AGI Architecture: Symbiosis of Subconscious and Conscious

文档状态：`REFERENCE`

> "意识只是冰山一角，巨大的潜意识在水面下支撑着一切。" —— 西格蒙德·弗洛伊德

## 1. 核心理念：重新定义"脑"与"心"

本项目不再是一个单纯的生态模拟器，而是一个**完整的 AGI 心理模型**。我们将智能系统划分为两个永不停息的进程：

1.  **潜意识 (The Subconscious) - Daerwen 3.5 Engine**
    *   **角色**：生物本能、直觉、快速反应、生命维持、梦境生成。
    *   **载体**：基于物理/化学/基因规则的 2D 生态系统 (`f:\avalanche-持续学习\daerwen3.5`).
    *   **运行模式**：**7x24小时永不停机**。它像人类的心跳和自主神经系统一样，在后台默默演化、处理海量信息、维持内稳态。
    *   **特性**：非语言的、混沌的、并行的、基于能量和梯度的。

2.  **表意识 (The Conscious) - LLM (Large Language Model)**
    *   **角色**：理性思考、语言表达、高级规划、自我反思、与人类交流。
    *   **载体**：接入的现代大模型（如 Gemini, GPT, Claude 等）。
    *   **运行模式**：**按需唤醒 (On-Demand / Attention-Based)**。当潜意识遇到处理不了的异常、或者积累了足够的模式需要总结时，"唤醒"表意识进行处理。
    *   **特性**：语言的、逻辑的、串行的、基于符号和意义的。

---

## 2. 交互接口：脑体桥梁 (Mind-Body Bridge)

这不仅仅是两个软件的连接，这是**生理信号与心理符号的转换**。

### 2.1 上行通道：从潜意识到表意识 (Bottom-Up: Sensation & Interoception)
*   **直觉/感受 (Feelings)**: Daerwen 引擎不仅输出原始数据，还需通过 `State Aggregator` 生成"感受信号"。
    *   *例如*：系统熵增过快 -> 输出 "焦虑 (Anxiety)" 信号。
    *   *例如*：资源极度丰富但物种单一 -> 输出 "空虚 (Boredom)" 信号。
*   **梦境 (Dreams)**: 潜意识在后台运行时的可视化快照（Image/Video）或压缩状态向量，作为素材喂给 LLM。
    *   *LLM 旁白*："我感觉到一阵莫名的躁动，底层的能量在涌动，似乎有一场变革正在酝酿..."

### 2.2 下行通道：从表意识到潜意识 (Top-Down: Intention & Will)
*   **意念/意志 (Will)**: LLM 不能直接控制每个粒子（正如你不能用意念控制心跳），但它可以施加**全局影响**。
    *   LLM 输出自然语言指令 -> 解析为 `ExternalInput` (环境参数/化学场)。
    *   *LLM 思考*："我们需要更多的多样性。" -> *Action*: 降低全域温度，在局部投放突变诱导剂。
*   **注意力 (Attention)**: LLM 的关注点会像探照灯一样，改变 Daerwen 引擎中某些区域的计算精度或资源分配。

---

## 3. 系统架构设计

```mermaid
graph TD
    subgraph "Subconscious (Daerwen 3.5)"
        A[Physics Engine] --> B[Chemistry Field]
        B --> C[Genetic Evolution]
        C --> D[Emergence Detector]
        D -- "Alerts / Patterns" --> E[State Aggregator]
    end

    subgraph "Interface (The Bridge)"
        E -- "Sensory Stream (JSON/Tensor)" --> F[Translator]
        F -- "Prompts / Context" --> G[Conscious Mind]
        H[Control Signal] -- "ExternalInput" --> A
    end

    subgraph "Conscious (LLM)"
        G[LLM Agent] -- "Reflection / Planning" --> I[Memory (Vector DB)]
        I --> G
        G -- "Intention" --> H
        G -- "Chat" --> User
    end
```

### 关键组件重构

1.  **Daerwen Engine (Server Mode)**: 
    *   必须改造为**无头服务 (Headless Service)**，可在后台长期运行。
    *   增加**持久化存储 (Long-term Memory)**：使用 Avalanche 库（或简单的数据库）记录漫长的进化历史，形成"深层记忆"。

2.  **The Translator (中间件)**:
    *   **Sensation Encoder**: 将生态系统的统计数据（熵、能量、多样性）编码为自然语言描述或 Embeddings。
    *   **Action Decoder**: 将 LLM 的模糊意图（"让世界更狂野一点"）解码为具体的物理参数调整（`mutation_rate += 0.05`）。

3.  **Avalanche Integration**:
    *   利用 `avalanche-lib` 实现**终身学习 (Continual Learning)**。
    *   潜意识不仅在运行，还在**学习**。它会训练一个轻量级的策略网络（Policy Network），处理日常琐事（如维持基本能量平衡），只有遇到没见过的灾难时才叫醒 LLM。

---

## 4. 实施路线图

1.  **Phase 1: 潜意识独立 (The Sleeping Body)**
    *   完善 `daerwen3.5`，使其能在服务器上长期稳定运行，不依赖前端打开。
    *   实现 `Avalanche` 库的初步集成，让潜意识具备基本的"条件反射"能力。

2.  **Phase 2: 建立连接 (The Awakening)**
    *   编写 `bridge.py`，连接 Daerwen Server 和 LLM API。
    *   定义"感受词汇表"：定义哪些物理状态对应哪些心理感受。

3.  **Phase 3: 共生进化 (Symbiosis)**
    *   让系统运行一周。
    *   LLM 定期写"观察日记"（基于潜意识的数据）。
    *   用户可以通过与 LLM 对话，间接影响那个正在进化的数字世界。

---

## 5. 哲学意义

*   **表意识是潜意识的解释器**：LLM 实际上是在为 Daerwen 系统中发生的混沌物理现象提供"合理化解释"（Rationalization），这正如人类大脑的工作方式。
*   **人机共生**：用户与 LLM 交流，LLM 与 Daerwen 交流。这是一个三层嵌套的智能系统。
