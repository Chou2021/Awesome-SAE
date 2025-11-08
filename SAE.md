# Sparse Autoencoders

多义性（polysemanticity）：神经元常在语义无关的语境中激活（如同一神经元既响应 “人名” 又响应 “数字”

叠加（superposition）导致多义性


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
- 

## 7. Scaling and evaluating sparse autoencoders (ICLR 2025)

## 8. Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models (ICLR 2025)

## 9. Efficient Dictionary Learning with Switch Sparse Autoencoders (ICLR 2025)

## 10. Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words (ICLR 2025)

## 11. Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models (ICLR 2025)

## 12. Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control (ICLR 2025)

## 13. Sparse Autoencoders Do Not Find Canonical Units of Analysis (ICLR 2025)

## 14. Sparse autoencoders reveal selective remapping of visual concepts during adaptation (ICLR 2025)

## 15. **Residual Stream Analysis with Multi-Layer SAEs (ICLR 2025)**

## 16. **SAE-V: Interpreting Multimodal Models for Enhanced Alignment (ICML 2025)**

## 17. SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders (ICML 2025)

## 18. Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models (ICML 2025)

## 19. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability (ICML 2025)

## 20. From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models (ICML 2025)

## 21. Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment (ICML 2025)

## 22. Learning Multi-Level Features with Matryoshka Sparse Autoencoders (ICML 2025)

## 23. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders (ICML 2025)

## 24. Scaling Sparse Feature Circuits For Studying In-Context Learning (ICML 2025)

## 25. Sparse Autoencoders, Again? (ICML 2025)

## 26. Low-Rank Adapting Models for Sparse Autoencoders (ICML 2025)

## 27. Sparse Autoencoders for Hypothesis Generation (ICML 2025)

## 28. Interpreting CLIP with Hierarchical Sparse Autoencoders (ICML 2025)

## 29. Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations (ICML 2025)

## 30. Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders (ICML 2025)

## 31. Are Sparse Autoencoders Useful? A Case Study in Sparse Probing (ICML 2025)

## 32. Constrain Alignment with Sparse Autoencoders (ICML 2025)

## 33. The Complexity of Learning Sparse Superposed Features with Feedback (ICML 2025)

## 34. Disentangling Superpositions: Interpretable Brain Encoding Model with Sparse Concept Atoms (NeurIPS 2025)

## 35. **One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models (NeurIPS 2025)**

## 36. **SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders (NeurIPS 2025)**

## 37. **Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders (NeurIPS 2025)**

## 38. **Sparse Diffusion Autoencoder for Test-time Adapting Prediction of Spatiotemporal Dynamics (NeurIPS 2025)**

## 39. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit (NeurIPS 2025)

## 40. Revising and Falsifying Sparse Autoencoder Feature Explanations (NeurIPS 2025)

## 41. Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry (NeurIPS 2025)

## 42. Proxy-SPEX: Sample-Efficient Interpretability via Sparse Feature Interactions in LLMs (NeurIPS 2025)

## 43. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders (NeurIPS 2025)

## 44. **Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models (NeurIPS 2025)**

## 45. **VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set (NeurIPS 2025)**

## 46. Dense SAE Latents Are Features, Not Bugs (NeurIPS 2025)

## 47. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders (ICCV 2025)

## 48. Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders (ACL 2025)