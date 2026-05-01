# Sentiment2Stock
Predicting Markets from Social Trends
## 👥 Team: Brain Rots
| Role | Registration Number | Name |
| :--- | :--- | :--- |
| **Member-1** | `23BIT0232` | PRITIKA KASINATHAN |
| **Member-2** | `23BIT0398` | BHARATH KUMAAR S K |
| **Member-3** | `23BIT0347` | VARSHA K |
| **Member-4** | `23BIT0096` | RAHUL R |
| **Member-5** | `23BIT0333` | HARINI SHREE S S |

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Details](#-dataset-details)
3. [Pipeline Architecture](#%EF%B8%8F-pipeline-architecture)
4. [Annotation & Labeling Guidelines](#-annotation--labeling-guidelines)
5. [Evaluation Metrics & Results](#-evaluation-metrics--results)

---

## 🎯 Project Overview
This project explores the intersection of social media sentiment and financial market movements. We developed a robust **Semi-Supervised Learning (SSL)** pipeline to classify and predict market behavior using text embeddings, emotional vectors, and engineered market indicators across 31 stock tickers and crypto assets (e.g., DOGE-USD, PEPE-USD).

The classification targets four distinct behavioral states:
* `[1, 1]` - **Availability + Herd:** News-driven hype (both fundamental catalyst and memetic social momentum).
* `[1, 0]` - **Pure Availability:** Purely news-driven reaction.
* `[0, 1]` - **Pure Herd:** Purely crowd-driven hype without recent news.
* `[0, 0]` - **Control:** Neither. Market noise / generic chatter.

---

## 📊 Dataset Details

### Data Sources
* **Social Media:** Over 80,000 posts covering tickers like RBLX, GPRO, SNAP, and crypto.
* **Financial Events:** Global economic indicators (GDP, CPI, central bank decisions).
* **Market Data:** 166,987 intraday bars (5-minute intervals) mapped via `yfinance` and CoinGecko API.

### Preprocessing & Consolidation
* **Initial Merged Data:** 98,089 records (August 27, 2025 – October 27, 2025).
* **Deduplication & Filtering:** Removed 18,087 duplicates and filtered out old posts. Kept records with `headline_similarity >= 0.65`.
* **Final Shape:** 80,002 rows × 514 columns.
* **Missing Data:** Carefully managed timestamp issues (100% valid `NaT`) and used timezone-aligned fallback strategies.

### Feature Engineering (514 Features)
Sophisticated behavioral metrics were engineered and z-score normalized:
* `A_Emotional_News_Delta`: Emotional divergence from a news baseline.
* `H_Social_Momentum_Ratio`: Social signal velocity.
* `H_Jargon_Sentiment_Homogeneity`: Technical terminology alignment.
* `H_Retail_Volume_Float_Rotation`: Volume-to-float ratio refinement.
* **Embeddings:** 384-dimensional Sentence Transformer emotion vectors (Neutral, Positive, Fear, Trust, Negative, etc.).

---

## ⚙️ Pipeline Architecture

Our Machine Learning architecture revolves around handling heavily imbalanced datasets and sparse labels:

1.  **Dimensionality Reduction:** Utilized **LightGBM** to execute embedded feature selection (split/gain importance), distilling 500+ features into a compact, high-signal subset without an external reducer.
2.  **Stacked Ensemble:** Soft-voting ensemble combining **LightGBM** and **CatBoost** over 5-fold cross-validation.
3.  **Semi-Supervised Learning (SSL) Cycles:**
    * **SSL-1:** Generated high-confidence pseudo-labels using rare-class quotas.
    * **SSL-2:** Retrained ensemble with expanded data + strict boundary guards.
    * **SSL-3:** Dynamically tuned thresholds + integrated k-NN proximity filtering.
4.  **Co-Training:** Exchanged confident pseudo-labels between complementary views (Social/Linguistic vs. Market/Temporal).
5.  **Label Propagation:** Sparse label propagation on a k-NN graph to map out decision boundaries and recover hidden minority structures.

---

## 📝 Annotation & Labeling Guidelines

For the gold standard validation and train seed update, we relied on manual annotation. Annotators acted as a "Mail Sorter," assigning posts to the four defined bins.

### The "Mail Sorter" Workflow
* **Top Slice (High Confidence):** These are the "easy" records. Expect 85-95% agreement with the model's `predicted_label`. Review efficiently.
* **Bottom Slice (Boundary/Edge cases):** These are the ambiguous records. Agreement will be lower (50-70%). This is where human intuition handles nuances the rule-based script missed.

### Detective Clue Checklist
* **[1,0] Availability:** High `headline_similarity_score`, low `time_offset_from_event`, low `crowd_keyword_count`.
* **[0,1] Herding:** High `sentiment_homogeneity`, high `crowd_keyword_count`, high `social_proof_velocity`.
* **[1,1] Both:** A perfect storm. Both news-driven metrics and herding metrics spike simultaneously.
* **[0,0] Control:** Background chatter. Low similarity, low engagement, low memetic language.

> ⚠️ **CRITICAL: AVOID TARGET LEAKAGE**
> Annotators MUST strictly ignore specific market outcome features during labeling to prevent model contamination. **Do not look at:**
> * `intraday_return_30min`
> * `intraday_return_24hr`
> * `intraday_volatility`
> * `volume_spike_ratio`

---

## 🏆 Evaluation Metrics & Results

The project aims for a target F1-Score of **> 0.75**, with an "Excellent" rating above **0.85**. Our advanced SSL and Co-Training methods significantly outperformed baselines.

| Pipeline Stage | Macro F1 | ROC AUC | Brier Score | MCC | Kappa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SSL-1** | 0.97160 | 0.99959 | 0.09199 | 0.96894 | 0.96878 |
| **SSL-2** | 0.96993 | 0.99946 | 0.06587 | 0.93559 | 0.93360 |
| **SSL-3** | 0.96041 | 0.99846 | 0.07505 | 0.90666 | 0.90401 |
| **CoTrain_FullGold** | **0.98344** | 0.99889 | 0.08068 | **0.97529** | **0.97529** |
| **Label_Propagation**| 0.95185 | 0.99185 | 0.05684 | 0.93428 | 0.93370 |

> **Conclusion:** The Co-Training model split along semantic views (Text/Behavior vs. Market Context) successfully generated the highest predictive power, peaking at a 0.983 Macro F1 Score.
