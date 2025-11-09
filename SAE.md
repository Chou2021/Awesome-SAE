# Sparse Autoencoders

多义性（polysemanticity）：神经元常在语义无关的语境中激活（如同一神经元既响应 “人名” 又响应 “数字”

叠加（superposition）导致多义性

## 流程
  - 介绍 Vanilla SAE
  - 侧重不同模型、不同任务上的应用（LLM、LVLM、Diffusion Model、Text-to-Image Model）


## 1. Sparse autoencoders find highly interpretable features in language models (ICLR 2024)
![插入图片](./Sparse%20autoencoders%20find%20highly%20interpretable%20features%20in%20language%20models/1.png)
- 使用 SAE 解释小型 LM
- SAE 架构设计：
  - 挑选输入层：语言模型内部激活（残差流、MLP 子层、注意力头子层）
  - 内部稀疏表征：$d_{\mathrm{hid}}=R\times d_{\mathrm{in}}$，ReLU 激活，实现超完备表征
  - 编码器和解码器权重绑定：编码器矩阵$M\in\mathbb{R}^{d_{\mathrm{hid}}\times d_{\mathrm{in}}}$，解码器矩阵$M^\top$，① 减半内存消耗；② 明确特征方向（避免编码器 / 解码器歧义）；③ 残差流训练无性能损失 $$\mathbf{c}=\mathrm{ReLU}(M\mathbf{x}+\mathbf{b})$$ $$\hat{\mathbf{x}}=M^\top\mathbf{c}=\sum_{i=1}^{d_{\mathrm{hid}}-1}c_i\mathbf{f}_i$$ $$\mathcal{L}(\mathbf{x})=\|\mathbf{x}-\hat{\mathbf{x}}\|_2^2+\alpha\|\mathbf{c}\|_1$$
  - 残差流$\alpha=8.6e-4$，MLP $\alpha=3.2e−4$
  - 对$M$进行行归一化，防止模型通过增大特征向量规模来降低稀疏损失
  - 自动可解释性评分：
### 评论：局限性之一 —— 重构损失非零

## 2. Improving Sparse Decomposition of Language Model Activations with Gated Sparse Autoencoders (NeurIPS 2024)
- 为鼓励稀疏性施加的 L1 惩罚会导致收缩偏差（系统低估特征激活幅度），需在重建保真度与稀疏性间妥协，损害分解准确性。
- 将传统 SAE 编码器的 “特征激活判断” 与 “幅度估计” 功能分离，仅对前者施加 L1 惩罚，减少偏差影响范围 $$\mathbf{\tilde{f}}(\mathbf{x})\coloneqq\mathbb{1}[(\mathbf{W}_{\mathrm{gate}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{gate}})>0]\odot\mathrm{ReLU}(\mathbf{W}_{\mathrm{mag}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{mag}})$$
- 为避免参数激增，采取权重共享方案
$$(\mathbf{W}_{\mathrm{mag}})_{ij}\coloneqq(\exp(\mathbf{r}_\mathrm{mag}))_i\cdot(\mathbf{W}_\mathrm{gate})_{ij}$$
- $$\mathcal{L}_\mathrm{gated}(\mathbf{x})\coloneqq\|\mathbf{x}-\hat{\mathbf{x}}(\mathbf{\tilde{f}}(\mathbf{x}))\|_2^2+\lambda\|\mathrm{ReLU}(\pi_\mathrm{gated}(\mathbf{x}))\|_1+\|\mathbf{x}-\hat{\mathbf{x}}_\mathrm{frozen}(\mathrm{ReLU}(\pi_\mathrm{gated}(\mathbf{x})))\|_2^2$$

## 3. Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE) (NeurIPS 2024)
- 提出**任务无关、无需训练**的方法，将 CLIP 嵌入分解为**稀疏、非负、人类可解释**的概念组合
- 稀疏分解的四个充分条件
  - 图文在概念空间稀疏，即潜在概念向量 $\omega$ 的非零元素数 $\|\omega\|_0\le \alpha$（$\alpha \ll \text{概念总数} k$）
  - CLIP 仅捕获语义概念，忽略非语义噪声（如光照、相机角度）
  - CLIP 在概念空间线性（符合 “线性表示假设”）
  - CLIP 图文编码器对齐（需预处理解决模态间隙）
  - 实验验证：将两张图片（或两段文本）拼接，对比组合嵌入与个体嵌入加权平均的相似度，发现权重接近**0.5**
- 方法
  - 选择 LAION-400m 文本标注中最频繁的 1-2 字短语（bigrams）；移除 NSFW 样本；剔除相似度 > 0.9 的概念；
  - 模态对齐预处理：图像嵌入和文本嵌入分布在单位球的 “不同锥面” 上 —— 图像 - 图像、文本 - 文本的跨样本余弦相似度集中于正值，而图像 - 文本的相似度集中于 0
    - 图像嵌入中心化：计算 MSCOCO 训练集所有图像嵌入的均值$\mu_{img}$，将待分解的图像嵌入减去该均值得到中心化嵌入 $z=\sigma(z^{img}-\mu_{img})$
    - 概念词典中心化：词典矩阵 $C$ 减去词典中所有概念嵌入的均值
    - 分解得到概念权重 $w^*$ 后，$\hat{z}^{img}=\sigma(Cw^*+\mu_{img})$
  - 稀疏性与语义重建的权衡 $$\min_{w\in\mathbb{R}_+^c}\|Cw-z\|_2^2+2\lambda\|w\|_1$$
### 评论：预先定义码本，不需要额外解释特征，但重建误差较大，可解释性局限于码本概念

## 4. Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning (NeurIPS 2024)
- 传统 SAE 因聚焦激活重构而优先学习数据集结构，忽略网络功能重要特征
- 通过最小化原始模型与插入 SAE 后模型的输出分布 KL 散度训练 SAE $$\mathcal{L}_\text{e2e+ds}=\text{KL}(\hat{y},y)+\phi\|Enc(a^{(l)})\|_1+\frac{\beta_l}{L-l}\sum_{k=l+1}^{L-1}\|\hat{a}^{(k)}-a^{(k)}\|_2^2$$
- 增加下游层重构损失：防止模型通过非原始计算路径实现输出分布匹配（如利用下游层未被原模型使用的变换）
### 评论：仍需要保证特征准确重构

## 5. Codebook features: Sparse and discrete interpretability for neural networks (ICML 2024)
![输入图片](./Codebook%20Features%20Sparse%20and%20Discrete%20Interpretability%20for%20Neural%20Networks/1.jpeg)
- 输入：隐藏层激活向量 $a\in\mathbb{R}^N$，码本 $C={c_1,c_2,...,c_C}\in\mathbb{R}^{C\times N}$
- 根据余弦相似度，用 top-k 个最相似码向量的和替换 $a$ (k 控制稀疏度)
- 基于预训练模型微调，在注意力头和/或 MLP 块插入码本，保留残差流完整性
- $$\mathcal{L}=-\sum_{i=1}^N\log p_\theta(x_i|x_{<i})+\lambda\cdot\text{MSE}(\mathcal{C}(x),\text{stop-gradient(x)})$$

### 评论：和之前的 end-to-end SAE 类似

## 6. Interpreting Attention Layer Outputs with Sparse Autoencoders (ICML 2024)
![输入图片](/Interpreting%20Attention%20Layer%20Outputs%20with%20Sparse%20Autoencoders/1.jpeg)
- 将 SAE 应用于注意力层输出（并非传统的 MLP 或残差流），揭示注意力层学习的核心概念
- 输入项：**Scaled Dot-Product Attention** 输出后拼接的、过线性层之前的输出 $\mathbf{z}_\text{cat}$
- 重构公式：$$\mathbf{z}_\text{cat}=\mathbf{\hat{z}_\text{cat}}+\epsilon(\mathbf{z}_{cat})=\sum_{i=0}^{d_\text{sae}}f_i(\mathbf{z}_\text{cat})\mathbf{d}_i+\mathbf{b}+\epsilon(\mathbf{z}_\text{cat})$$ $\epsilon(\mathbf{z}_\text{cat})$ 是误差项
- 归因技术
  - 基于权重的头归因：将特征方向 $\mathbf{d}_i$ 拆分为各头的子向量 $\mathbf{d}_{i, j}$，计算头 $k$ 的贡献占比 $h_{i, k}=\frac{\|\mathbf{d}_{i,k}\|_2}{\sum_{j=1}^{n_\text{heads}}\|\mathbf{d}_{i,j}\|_2}$
  - 直接特征归因（DFA）：按照头位置分解 SAE 预激活 $f_i^{\text{pre}}(\mathbf{z}_\text{cat})=\mathbf{w}_i^\top\mathbf{z}_\text{cat}=\mathbf{w}_{i,1}^\top\mathbf{z}_1+\mathbf{w}_{i,2}^\top\mathbf{z}_2+\cdots+\mathbf{w}_{i,n_\text{heads}}^\top\mathbf{z}_{n_\text{heads}}$
  - 递归 DFA：冻结注意力模式 / LayerNorm，线性追溯特征至上游残差流、组件及早期 token
- 结论
  - 多义性：对 GPT-2 Small 仅发现 14 个单义候选头（Top10 归因特征高度相关），至少 90% 的头存在多义性
  - 层间功能规律：
    - 早期层（0-3）：语法特征（单 token、bigram）+ 简单实体跟踪；
    - 中期层（4-9）：复杂语义（动词家族、推理短语、时间关系）；
    - 晚期层（10-11）：语法调整、bigram 补全 + 少量长程上下文跟踪。
### 评论：未覆盖 Transformer 的 QK 电路等关键组件；SAE 的线性分解可能无法捕捉注意力层的非线性计算


## 7. Scaling and evaluating sparse autoencoders (ICLR 2025)
- 重建与稀疏性目标难以平衡，需调优L1正则系数；大规模训练中死潜变量多；现有研究仅聚焦小规模 SAE
- **k-Sparse SAE**
  - 编码器仅保留前 k 个最大潜变量激活，解码器不变
  - 编码器 - 解码器转置初始化
  - 利用死潜变量重建误差，使死潜变量获得梯度信号 $\mathcal{L}_\text{aux}=\|(\mathbf{x}-\hat{\mathbf{x}})-\mathbf{W}_\text{dec}\mathbf{z}_\text{dead}\|_2^2$


## 8. Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models (ICLR 2025)
- 粗粒度组件分析具有多义性；基于神经元、线性探测的细粒度方法依赖人工设计标注


## 9. Efficient Dictionary Learning with Switch Sparse Autoencoders (ICLR 2025)
![输入图片](./Efficient%20Dictionary%20Learning%20with%20Switch%20Sparse%20Autoencoders/1.png)
- 密集编码器前向传播是时间瓶颈，潜变量预激活存储是内存瓶颈
- 路由网络 + 多专家 SAE
  - 仅激活一个专家


## 10. Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words (ICLR 2025)
![](./Rethinking%20Evaluation%20of%20Sparse%20Autoencoders%20through%20the%20Representation%20of%20Polysemous%20Words/1.png)
- 现有评估无法直接验证 SAE 是否实现 “提取单义特征” 的核心目标。提出聚焦多义词的 SAE 评估框架 **PS-Eval**。

## 11. Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models (ICLR 2025)
- “仅训练下一个 token 预测的模型为何能实现 RL” 的机制未知
- 验证 LLM 是否自发涌现时序差分（ TD ）学习机制。

## 12. Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control (ICLR 2025)
![](./Towards%20Principled%20Evaluations%20of%20Sparse%20Autoencoders%20for%20Interpretability%20and%20Control/1.png)
- SAE 缺少 Ground Truth 导致评估困难
  - 任务属性参数化：选择任务相关的、人类可理解的属性（如 IO 任务的 IO/S/Pos），确保属性能完全捕捉任务关键信息。
  - 构建监督特征字典
  - 无监督字典评估：以监督字典为基准，从三个维度（重建的必要性 / 充分性、属性稀疏可控性、因果一致性可解释性）评估 SAE，且评估过程不依赖特征的人工解释（如可控性测试通过优化特征子集实现属性编辑，无需知道特征对应哪个属性）。

## 13. Sparse Autoencoders Do Not Find Canonical Units of Analysis (ICLR 2025)
- **Motivation**
  - 机械可解释性的核心诉求：将 LLM 的激活分解为 “规范分析单元”—— 即满足**唯一性**（无变体）、**完整性**（覆盖所有必要特征）、**原子性**（不可再分）的可解释特征。
  - 此前假设“足够大的 SAE 字典能够找到规范单元”未被验证
- 验证完整性：通过在不同字典大小的 SAE 间插入 / 替换潜变量，观察重构性能变化，以分类大 SAE 的潜变量类型。
- 验证原子性：**meta-SAEs**
  - meta-SAEs 的训练数据并非 LLM 激活，而是另一 SAE 的解码器矩阵 $\mathbf{W_\text{dec}}$ —— 即把大 SAE 的每个潜变量（对应 $\mathbf{W_\text{dec}}$ 的一列）作为训练样本，目标是学习对这些潜变量的稀疏分解。
- 结论：小 SAE 不完整，大 SAE 非原子性

## 14. **Sparse autoencoders reveal selective remapping of visual concepts during adaptation (ICLR 2025)**
- CLIP 通过提示适配可以高效适配下游任务，需要研究其内部表征在适配过程中如何变化；提出针对 CLIP ViT 的 Patch-SAE，能够提取细粒度（如形状、颜色、语义）的可解释视觉概念及补丁级空间归因；分析 CLIP 在分类任务中的行为
- **Patch-SAE**：包含图像 token 的 SAE ![](./Sparse%20Autoencoders%20Reveal%20Selective%20Remapping%20of%20Visual%20Concepts%20during%20Adaptation/1.png)
  - 输入：冻结 CLIP ViT-B/16 的第 11 层（倒数第二层）残差流输出，包含 1 个 [CLS] token + 14×14=196 个图像 token，每个 token 维度 $d_\text{ViT}=768$。
  - 编码器：$W_E \in \mathbb{R}^{d_\text{ViT}\times d_\text{SAE}}$
  - 激活函数：$\phi$ ReLU
  - 解码器：$W_D\in\mathbb{R}^{d_\text{SAE}\times d_\text{ViT}}$ $$\text{SAE}(\mathbf{z})=W_D^\top\phi(W_E^\top\mathbf{z})$$ $$\mathcal{L}_\text{SAE}=\|\text{SAE}(\mathbf{z})-\mathbf{z}\|_2^2+\lambda_{l_1}\|\phi f(\mathbf{z})\|_1$$



- **分析 SAE 的 latents**
![](./Sparse%20Autoencoders%20Reveal%20Selective%20Remapping%20of%20Visual%20Concepts%20during%20Adaptation/2.png)
  - 对于每个 SAE 的潜变量（共 $d_\text{SAE}$ 个），将能最大程度激活该 SAE 潜变量的 top-k 图像作为参考图像（共 $d_\text{SAE}\times k$ 张图像）
  - 计算激活分布的汇总统计
    - **Sparsity**：表示该 latent 被激活的频率。高频的 latent 可能代表一个常见概念或一个无法解释的噪声
    - **Mean activation value**：对激活样本中的正激活值取平均值来计算。反应了 SAE 的置信度。如果一个 latent 有比较高的平均激活值，则它更有可能代表一个有意义的概念
    - **Label entropy**：衡量有多少个不同的标签激活了 latent。熵为零表示所有参考图像都具有相同的标签。熵值越高代表有更多标签对 latent 的激活有贡献。
    - **Label standard deviation**
  - 将 patch-level 激活转化为图像级激活、类别级激活、数据集级激活
    - 使用一个较小的阈值 $\tau$，对第 i 张图的第 j 个 token 的激活向量二值化 $$\mathbf{a}_{i,j}[s]=\mathbb{I}(\mathbf{h}_{i,j}[s]>\tau),\text{ where }1\le s\le d_\text{SAE}$$ $$\mathbf{a}_i[s]=\sum_{j=1}^{n_i}\mathbf{a}_{i,j}[s],\ \mathbf{a}_c[s]=\sum_{i\in\mathcal{I}_c}\mathbf{a}_i[s],\ \mathbf{a}_D[s]=\sum_{i\in D}\mathbf{a}_i[s]$$
  - Localizing
    - 对于第 s 个 latent 和图像 $x_i$，利用激活值 $\mathbf{h}_{i,j}[s]$ 可以在图像中突出显示对应概念
![](./Sparse%20Autoencoders%20Reveal%20Selective%20Remapping%20of%20Visual%20Concepts%20during%20Adaptation/3.jpeg)

- **研究 SAE latents 与分类任务下模型行为之间的关系**
  - 将 CLIP 图像编码器的**中间层表示**替换为 SAE 重构的输出，观察性能 —— SAE latents 包含类别判别信息
  - CLIP 和 MaPLe 中激活度最高的 SAE latents 大多重叠
  - CLIP 和 MaPLe 中 top 的 SAE latents 影响不同：MaPLE 比 CLIP 更有效地利用了相同数量的 SAE 潜在变量进行分类
  - 基于提示的 Adaptation 通过优化常见激活概念与下游任务类别之间映射关系，带来性能提升

### 评论：有许多高标签熵的 latent 无法解释

## 15. Residual Stream Analysis with Multi-Layer SAEs (ICLR 2025)
- 传统 SAE 仅在单个 Transformer层的激活向量上训练，无法研究信息在跨层残差流中的流动与变化；现有研究推测模型可能**通过多层同时激活编码语义概念**，单层 SAE 无法捕捉该特征
- 基于残差流理论：Transformer 通过自注意力和 MLP 层选择性读写残差流信息，且相邻层残差流向量应具有较高相似性
- **MLSAE** 在所有层的残差流激活向量上训练，且参数跨层共享


## 16. **SAE-V: Interpreting Multimodal Models for Enhanced Alignment (ICML 2025)**

## 17. **SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders (ICML 2025)**

## 18. Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models (ICML 2025)

## 19. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability (ICML 2025)

## 20. From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models (ICML 2025)
- 略

## 21. Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment (ICML 2025)

## 22. Learning Multi-Level Features with Matryoshka Sparse Autoencoders (ICML 2025)

## 23. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders (ICML 2025)

## 24. **Scaling Sparse Feature Circuits For Studying In-Context Learning (ICML 2025)**

## 25. Sparse Autoencoders, Again? (ICML 2025)

## 26. Low-Rank Adapting Models for Sparse Autoencoders (ICML 2025)

## 27. Sparse Autoencoders for Hypothesis Generation (ICML 2025)

## 28. **Interpreting CLIP with Hierarchical Sparse Autoencoders (ICML 2025)**

## 29. Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations (ICML 2025)

## 30. Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders (ICML 2025)

## 31. Are Sparse Autoencoders Useful? A Case Study in Sparse Probing (ICML 2025)

## 32. Constrain Alignment with Sparse Autoencoders (ICML 2025)

## 33. The Complexity of Learning Sparse Superposed Features with Feedback (ICML 2025)

## 34. **SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders (ICCV 2025)**

## 35. Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders (ACL 2025)

## 36. Disentangling Superpositions: Interpretable Brain Encoding Model with Sparse Concept Atoms (NeurIPS 2025)
- 略

## 37. **One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models (NeurIPS 2025)**

## 38. **SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders (NeurIPS 2025)**

## 39. **Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders (NeurIPS 2025)**

## 40. **Sparse Diffusion Autoencoder for Test-time Adapting Prediction of Spatiotemporal Dynamics (NeurIPS 2025)**

## 41. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit (NeurIPS 2025)

## 42. Revising and Falsifying Sparse Autoencoder Feature Explanations (NeurIPS 2025)

## 43. Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry (NeurIPS 2025)


## 44. Proxy-SPEX: Sample-Efficient Interpretability via Sparse Feature Interactions in LLMs (NeurIPS 2025)

## 45. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders (NeurIPS 2025)

## 46. **Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models (NeurIPS 2025)**

## 47. **VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set (NeurIPS 2025)**

## 48. Dense SAE Latents Are Features, Not Bugs (NeurIPS 2025)

