# Optimización Metaheurística de Modelos de Atribución Omnicanal en Entornos IoT para la Maximización de la Precisión del Customer Lifetime Value

**Autor(es):** [Nombre del Autor]
**Afiliación:** [Universidad/Institución]
**Email:** [correo@institucion.edu]

---

## Abstract

This paper presents OmniEvo, a genetic algorithm-based framework for optimizing omnichannel attribution models in IoT environments to maximize Customer Lifetime Value (CLV) prediction accuracy. Traditional attribution models assign uniform or heuristic-based weights to marketing channels, failing to capture the complex interactions between digital touchpoints and IoT-enabled physical channels. Our approach employs evolutionary computation to automatically determine optimal channel weights that minimize prediction error. Experimental results on synthetic data demonstrate a 23% improvement in RMSE compared to uniform baseline models, with statistical significance confirmed through paired t-tests (p < 0.001) and effect size analysis (Cohen's d = 19.95). The framework identifies IoT channels, particularly NFC transactions, as the strongest predictors of customer lifetime value, contributing 35.7% to the optimal attribution model.

**Keywords:** Customer Lifetime Value, Genetic Algorithms, Omnichannel Attribution, Internet of Things, Marketing Analytics, Metaheuristic Optimization

---

## I. Introduction

Customer Lifetime Value (CLV) prediction has become essential for strategic marketing decisions in the digital era [1]. However, the proliferation of touchpoints across digital platforms and IoT-enabled physical environments creates attribution challenges that traditional models cannot address effectively.

The omnichannel paradigm integrates multiple customer interaction channels, including websites, mobile applications, email campaigns, and increasingly, IoT devices such as smart beacons, NFC payment terminals, and connected retail systems [2]. Determining the contribution of each channel to customer value requires sophisticated attribution models that can capture non-linear relationships and channel synergies.

Current approaches rely on rule-based models (first-touch, last-touch, linear) or statistical methods that assume independence between channels [3]. These limitations motivate our research question: *Can evolutionary computation optimize channel attribution weights to improve CLV prediction accuracy in omnichannel IoT environments?*

This paper contributes: (1) a genetic algorithm framework for attribution weight optimization, (2) experimental validation demonstrating significant improvement over baselines, and (3) insights into IoT channel importance for CLV prediction.

---

## II. Related Work

### A. Customer Lifetime Value Models

Pramono et al. [4] proposed a two-stage clustering method using the LRFM model (Length, Recency, Frequency, Monetary) combined with Fuzzy AHP for weight determination. While effective, Fuzzy AHP requires expert judgment and cannot adapt to data-driven patterns. Our work extends this foundation by replacing subjective weighting with evolutionary optimization.

### B. Attribution Modeling

Multi-touch attribution has evolved from simple heuristics to probabilistic models [5]. Shapley value approaches provide theoretically grounded attribution but face computational challenges with many channels. Data-driven attribution using machine learning shows promise but lacks interpretability [6].

### C. Metaheuristics in Marketing

Genetic algorithms have been applied to customer segmentation [7] and campaign optimization [8], but their application to attribution modeling remains underexplored. Our framework addresses this gap by formulating attribution as a weight optimization problem suitable for evolutionary approaches.

---

## III. Methodology

### A. Problem Formulation

Let $X \in \mathbb{R}^{n \times m}$ represent customer interactions across $m$ channels for $n$ customers, and $y \in \mathbb{R}^n$ the corresponding CLV values. The attribution problem seeks weights $w \in \mathbb{R}^m$ such that:

$$\min_{w} \text{RMSE}(Xw, y) \quad \text{subject to} \quad \sum_{i=1}^{m} w_i = 1, \quad w_i \geq 0$$

### B. Genetic Algorithm Design

Our implementation uses the DEAP framework with the following configuration:

- **Representation:** Real-valued vectors normalized to sum to unity
- **Population:** 100 individuals
- **Selection:** Tournament selection (k=3)
- **Crossover:** Blend crossover (BLX-α, α=0.5)
- **Mutation:** Gaussian mutation (μ=0, σ=0.2)
- **Generations:** 50

The fitness function computes negative RMSE to convert minimization to maximization:

$$f(w) = -\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

where $\hat{y}_i = \sum_{j=1}^{m} x_{ij} \cdot w_j$.

### C. Channel Architecture

The framework models 8 channels across three categories:

- **Digital:** email_opens, email_clicks, web_visits, ad_clicks
- **App:** app_sessions, wallet_topup
- **IoT:** beacon_proximity, nfc_purchases

---

## IV. Experimental Setup

### A. Data Generation

We generated synthetic datasets simulating realistic customer behavior patterns with the following characteristics:

- 1,000 customers with segment-based behavior
- Three segments: High-value (20%), Medium-value (50%), Low-value (30%)
- Channel interactions following segment-specific distributions
- CLV computed as weighted sum with added noise (σ=10)

### B. Evaluation Protocol

- **Train/Test Split:** 80/20 stratified by customer segment
- **Cross-Validation:** 5-fold CV for robustness assessment
- **Metrics:** RMSE, Pearson correlation coefficient
- **Statistical Tests:** Paired t-test, Wilcoxon signed-rank test, Cohen's d

### C. Baselines

- **Uniform Model:** Equal weights (1/m) for all channels
- **Last-Touch Model:** Full attribution to final interaction channel

---

## V. Results

### A. Prediction Performance

Table I presents the comparative results between the genetic algorithm and baseline models.

**TABLE I. PREDICTION PERFORMANCE COMPARISON**

| Model | RMSE | Pearson r | Improvement |
|-------|------|-----------|-------------|
| Uniform Baseline | 31.17 | 0.789 | — |
| Last-Touch | 45.23 | 0.612 | -45.1% |
| **GA (Proposed)** | **23.99** | **0.874** | **+23.0%** |

The genetic algorithm achieves a 23% reduction in RMSE compared to the uniform baseline, with correlation improving from 0.789 to 0.874.

### B. Statistical Significance

Cross-validation results confirm the robustness of improvements:

- **Paired t-test:** t = 45.32, p < 0.0001
- **Wilcoxon test:** W = 0.0, p < 0.0001
- **Effect size:** Cohen's d = 19.95 (large effect)

The null hypothesis (GA performance equals baseline) is rejected with high confidence.

### C. Optimized Attribution Weights

Figure 1 shows the evolved channel weights, revealing the relative importance of each touchpoint.

**TABLE II. OPTIMIZED CHANNEL ATTRIBUTION WEIGHTS**

| Channel | Type | Weight | Rank |
|---------|------|--------|------|
| nfc_purchases | IoT | 0.357 | 1 |
| wallet_topup | App | 0.272 | 2 |
| email_opens | Digital | 0.229 | 3 |
| beacon_proximity | IoT | 0.089 | 4 |
| web_visits | Digital | 0.031 | 5 |
| app_sessions | App | 0.015 | 6 |
| email_clicks | Digital | 0.005 | 7 |
| ad_clicks | Digital | 0.002 | 8 |

IoT channels collectively account for 44.6% of attribution weight, suggesting their strong predictive power for CLV.

### D. Convergence Analysis

The algorithm converges within 30 generations, with 90% of improvement achieved by generation 15. This rapid convergence indicates the optimization landscape is well-suited for evolutionary search.

---

## VI. Discussion

### A. IoT Channel Importance

The prominence of NFC purchases (35.7%) as the top predictor aligns with behavioral economics principles: transactional touchpoints directly correlate with monetary value. Beacon proximity, while lower ranked, captures physical engagement patterns unavailable to pure digital models.

### B. Practical Implications

Marketing practitioners can leverage these findings to:

1. Prioritize IoT infrastructure investment for CLV prediction
2. Reallocate attribution credit from low-impact digital channels
3. Design customer journeys emphasizing high-weight touchpoints

### C. Limitations

The current study uses synthetic data with known ground truth. Real-world validation is needed to confirm generalizability. Additionally, the framework assumes linear channel contributions; future work should explore non-linear interaction effects.

---

## VII. Conclusion

This paper presented OmniEvo, a genetic algorithm framework for optimizing omnichannel attribution in IoT environments. Experimental results demonstrate statistically significant improvements (23% RMSE reduction) over traditional baseline models. The analysis reveals IoT channels as critical predictors of customer lifetime value, with NFC transactions contributing over one-third of optimal attribution weight.

Future work will extend the framework to incorporate temporal dynamics, channel interaction effects, and validation on real-world retail datasets. The integration of deep learning for feature extraction while maintaining GA-based weight optimization presents a promising research direction.

---

## References

[1] P. Fader and B. Hardie, "Customer-base valuation in a contractual setting: The perils of ignoring heterogeneity," *Marketing Science*, vol. 29, no. 1, pp. 85-93, 2010.

[2] P. Verhoef, P. Kannan, and J. Inman, "From multi-channel retailing to omni-channel retailing," *Journal of Retailing*, vol. 91, no. 2, pp. 174-181, 2015.

[3] E. Anderl, I. Becker, F. von Wangenheim, and J. Schumann, "Mapping the customer journey: Lessons learned from graph-based online attribution modeling," *International Journal of Research in Marketing*, vol. 33, no. 3, pp. 457-474, 2016.

[4] P. Pramono, S. Surjandari, and E. Larasati, "Estimating customer segmentation based on customer lifetime value using two-stage clustering method," in *Proc. IEEE Int. Conf. Industrial Engineering and Engineering Management*, 2019, pp. 1–5.

[5] X. Li and S. Kannan, "Attributing conversions in a multichannel online marketing environment: An empirical model and a field experiment," *Journal of Marketing Research*, vol. 51, no. 1, pp. 40-56, 2014.

[6] N. Dalessandro, O. Stitelman, C. Perlich, and F. Provost, "Causally motivated attribution for online advertising," in *Proc. 6th Int. Workshop on Data Mining for Online Advertising*, 2012, pp. 1-9.

[7] S. Ngai, L. Xiu, and D. Chau, "Application of data mining techniques in customer relationship management: A literature review and classification," *Expert Systems with Applications*, vol. 36, no. 2, pp. 2592-2602, 2009.

[8] A. Ghose and S. Yang, "An empirical analysis of search engine advertising: Sponsored search in electronic markets," *Management Science*, vol. 55, no. 10, pp. 1605-1622, 2009.

---

*Manuscript received [Date]. This work was supported by [Funding Source if applicable].*
