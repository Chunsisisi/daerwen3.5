# 基因到显性的第一性原理文档

## 目的

回答核心问题：生物学主流研究中，基因如何通过物理/化学/生物过程产生显性性状，并将该逻辑映射到 DAERWEN 的可执行工程路线。

---

## 一、主流共识：不是“基因直接决定行为”

主流发育生物学与分子生物学的结论是：

- 基因不直接输出最终行为或形态；
- 基因通过多层调控和多尺度动力学，逐步生成可观察表型；
- 表型是网络动态结果，不是单一参数映射。

可压缩为链路：

`Genotype -> Gene Regulation -> Cell State Dynamics -> Patterning/Morphogenesis -> Biomechanics -> Phenotype`

---

## 二、从基因到显性的六层机制

### 1) 基因层（Genotype）

- 突变、重组、复制误差提供可遗传差异。

### 2) 调控层（Gene Regulation / GRN）

- 转录因子、增强子、抑制子、染色质可及性决定“哪些基因在何时何地表达”。
- 同一序列在不同调控状态可产生不同表达结果。

### 3) 细胞状态层（Cell State Dynamics）

- 转录/翻译、信号通路、代谢耦合形成动态细胞状态。
- 细胞状态不是静态值，而是连续时间系统。

### 4) 模式形成层（Patterning / Morphogenesis）

- 形态素梯度、反应-扩散、细胞间通讯共同形成空间模式。
- 发育中的边界和结构来自网络动力学，不是简单阈值开关。

### 5) 力学层（Biomechanics）

- 细胞增殖、张力、压缩、黏附与组织流变反过来改变信号与表达。
- 机械反馈是发育不可缺失的一环。

### 6) 显性层（Phenotype）

- 形态、生理、行为是前述各层在时间和空间上的综合结果。

---

## 三、第一性逻辑（物理 + 生物）

### 1) 自组织

- 局部相互作用可在无中心控制下形成宏观结构。

### 2) 非线性

- 反馈环、饱和效应、阈值和耦合会放大小差异。

### 3) 多尺度耦合

- 分子、细胞、组织三个尺度互相约束并相互回写。

### 4) 化学-力学共驱

- 化学信号决定材料状态，力学状态反过来改变表达与分化。

结论：

- “基因直解行为参数”不符合主流机制；
- “基因驱动生长与结构形成，再涌现行为”更符合第一性。

---

## 四、对 DAERWEN 的直接启示

### 当前主要偏差（需逐步替换）

- 当前主链仍较接近 `genome -> behavior parameters`。
- 扰动在不少场景承担了“制造复杂性”的主要角色。

### 目标链路（建议主线）

- `genome -> regulatory_state -> reaction_coefficients -> growth/structure -> behavior`

### 不该做的事

- 不再新增行为层闸门（如人工比例上限、年龄门槛）来“修稳定”。
- 不把高层目标分数直接写进底层更新规则。

### 应该做的事（最小改造顺序）

1. 先替换一个直连通道：行为阈值 -> 调控状态变量。
2. 再把调控状态映射到反应系数，不直接映射行为。
3. 让复制/交互由局部化学可达性和反应动力学决定。
4. 每次只改一个底层环节，并用固定 seed 协议回归验证。

---

## 五、权威参考（主流/可追溯）

### 教材与综述

- Alberts et al., *Molecular Biology of the Cell*（分子细胞生物学权威教材）
  - https://www.ncbi.nlm.nih.gov/books/NBK21054/
- Barresi & Gilbert, *Developmental Biology*（发育生物学经典教材）
- Carlberg, *Gene Regulation and Epigenetics*（2024）

### 形态素与模式形成

- Briscoe & Small (2015), *Development*, Morphogen rules
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4712844/
- Kicheva & Briscoe (2023), *Annual Review of Cell and Developmental Biology*
  - https://www.annualreviews.org/content/journals/10.1146/annurev-cellbio-020823-011522

### 反应-扩散与理论基础

- Turing (1952), *The Chemical Basis of Morphogenesis*
  - https://royalsocietypublishing.org/rstb/article/237/641/37/
- Meinhardt (2012), *Interface Focus*
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3363033/

### 力学生物学

- Corson et al. (2024), *Nature*（组织自组织力学）
- *Nature Cell Biology* (2024), 增殖驱动压缩与信号中心形成

---

## 六、作为项目决策的最短准则

- 凡是 `gene -> behavior` 直接映射，都默认视为过渡实现。
- 凡是能引入 `regulatory_state` 并保持可审计的改动，优先级更高。
- 凡是用上层目标去改底层规则的做法，默认视为设计师陷阱高风险。
