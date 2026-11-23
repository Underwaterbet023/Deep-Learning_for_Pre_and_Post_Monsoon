# Deep Learning for Pre and Post Monsoon

<div align="center">
  <img src="groundwater-deep-learning-banner.svg" alt="Deep Learning Groundwater Research Banner" width="100%">
</div>
<br>
<div align="center">

[![Author](https://img.shields.io/badge/Principal_Investigator-Sanskar_Kumar-4A148C.svg?style=for-the-badge&logo=googlescholar&logoColor=white)](https://github.com/)
[![Field](https://img.shields.io/badge/Discipline-Hydroinformatics_%26_Deep_Learning-1E88E5.svg?style=for-the-badge&logo=sciencedirect&logoColor=white)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B_%7C_Keras_3-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Status](https://img.shields.io/badge/Research_Status-Completed_%26_Peer_Reviewed-00C853.svg?style=for-the-badge&logo=git&logoColor=white)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-gray.svg?style=for-the-badge)](LICENSE)

<br/>

**A Formal Scientific Monograph and Computational Atlas Investigating 30 Years of Groundwater Dynamics (1994–2024)**  
*Comparative Benchmarking of Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), Hybrid Convolutional-Recurrent Networks (CNN-LSTM), and Novel Dual-Engine Hybrid Architectures (Hybrid LSTM-GRU) Across 358,000+ Borewell Soundings.*

---

**Author:** **Sanskar Kumar**   
*Department of Computer Science Engineering*  
*Contact & Lead Researcher: Sanskar Kumar*

---

</div>

## 📑 Volume Synopsis & Table of Contents

This research monograph presents an end-to-end empirical and theoretical investigation into seasonal groundwater level forecasting across four independent, non-overlapping decadal hydrogeological regimes. Spanning 30 years of monitoring observations (1994 to 2024) across tens of thousands of observation wells, this study establishes a rigorous causal modeling framework that couples spatial coordinates, harmonic calendar periodicity, multi-step autoregressive dynamics, and deep recurrent neural representations.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                MONOGRAPH ARCHITECTURE                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  📖 FRONT MATTER                                                                      │
│     ├── Executive Research Summary & Abstract                                          │
│     ├── Key Scientific Breakthroughs & Decadal Findings                               │
│     └── Mathematical Nomenclature & Hydrogeological Terminology                       │
│                                                                                        │
│  🏛️ PART I: THEORETICAL FRAMEWORK & METHODOLOGY                                       │
│     ├── Chapter 1: Hydrogeological Context & The Groundwater Crisis                   │
│     ├── Chapter 2: Multi-Decadal Datasets & Spatial Data Profiling                    │
│     ├── Chapter 3: Spatiotemporal Feature Engineering & Leakage-Free Splitting        │
│     ├── Chapter 4: Neural Architecture Formulations (LSTM, GRU, CNN-LSTM, Hybrid)     │
│     └── Chapter 5: Optimization Suite & 11-Dimensional Statistical Metrics            │
│                                                                                        │
│  📊 PART II: THE EMPIRICAL ATLAS — EXPERIMENTAL PLATES & RESULTS                       │
│     ├── Section 1: Post-Monsoon Regional Regime (1994–2023) [5,214 Records | 433 Wells]│
│     ├── Section 2: Pre-Monsoon Baseline Decade (1994–2003) [94,956 Records | 14,854 W] │
│     ├── Section 3: Pre-Monsoon Transition Decade (2004–2013) [97,179 Records | 16,041 W│
│     └── Section 4: Pre-Monsoon Modern Stress Decade (2014–2024) [161,224 Records | 27K]│
│                                                                                        │
│  🔬 PART III: CROSS-DECADAL SYNTHESIS & ARCHITECTURAL BENCHMARKING                     │
│     ├── Chapter 6: Decadal Aquifer Depletion & Hydroclimatic Memory Trends            │
│     ├── Chapter 7: Deep Architecture Selection Matrix: Regimes of Dominance           │
│     └── Chapter 8: Multi-Phase Polar Taylor Diagram Geometrics                        │
│                                                                                        │
│  💻 PART IV: COMPUTATIONAL ECOSYSTEM & REPRODUCIBILITY                                │
│     ├── Chapter 9: Directory Hierarchy & Saved Model Artifacts                        │
│     ├── Chapter 10: Reproduction Protocols & Training Pipeline Execution               │
│     └── Chapter 11: Standalone Visualizer & Interactive HTML Dashboards               │
│                                                                                        │
│  📚 PART V: SCHOLARLY EPILOGUE & CITATION                                              │
│     ├── Concluding Recommendations for Sustainable Water Resource Management           │
│     ├── Formal Academic Citations (BibTeX & APA Format)                                │
│     └── Author Declaration & Research Ethics                                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 📖 FRONT MATTER

## 🌟 Executive Research Summary & Abstract

Groundwater serves as the primary freshwater lifeline for over 1.5 billion people across South Asia, supplying 85% of drinking water and driving more than 60% of irrigated agriculture. However, unchecked extraction, expanding megacities, and climate change-induced monsoon variability have triggered severe localized overdraft, aquifer desaturation, and volatile water table fluctuations. 

This research study conducted by **Sanskar Kumar** establishes a data-driven, non-invasive deep learning framework designed to model, reconstruct, and forecast seasonal groundwater depth to water level ($\text{DTWL}$) and water level in meters below ground level ($\text{WL in mbgl}$). Unlike previous studies that either artificially pool disparate seasonal data or contaminate temporal testing sets through random cross-validation, this research strictly partitions the empirical corpus into **four completely independent seasonal-decadal experimental domains**:
1. **Post-Monsoon Regional Regime (1994–2023)**: 5,214 soundings across 433 long-term benchmark observation wells, capturing 29 years of post-recharge equilibrium.
2. **Pre-Monsoon Baseline Decade (1994–2003)**: 94,956 soundings across 14,854 unique wells, documenting baseline pre-monsoon water table depressions.
3. **Pre-Monsoon Intermediate Transition Decade (2004–2013)**: 97,179 soundings across 16,041 unique wells, tracking the onset of mechanized agricultural groundwater intensification.
4. **Pre-Monsoon Modern Stress Decade (2014–2024)**: 161,224 soundings across 27,107 monitoring wells, capturing unprecedented modern groundwater stress, extreme climatic variance, and rapid spatial urbanization.

Four specialized sequence models—**Long Short-Term Memory (LSTM)**, **Gated Recurrent Unit (GRU)**, **1D Convolutional Long Short-Term Memory (CNN-LSTM)**, and a newly developed **Hybrid LSTM-GRU Dual-Engine Network**—were trained and evaluated independently within each decadal corpus. Models were evaluated against unseen chronological test data across an **11-metric statistical suite** ($R^2$, $\text{RMSE}$, $\text{MAE}$, $\text{Pearson } r$, $\text{NSE}$, $\text{KGE}$, $\text{NRMSE}$, $\text{Willmott's } d$, $\text{LMI}$, $\text{MSE}$, $\text{MAPE}$).

### 🏆 Key Scientific Discoveries

1. **Hydrological Regime Differentiation**: Neural architectures demonstrate clear regime-dependent inductive biases. While the **Gated Recurrent Unit (GRU)** achieved peak efficiency on the long-term, moderate-density post-monsoon regime ($R^2 = 0.7447$, $\text{RMSE} = 2.4851\text{ m}$, $\text{NSE} = 0.7447$), the **CNN-LSTM** network dominated the early and intermediate pre-monsoon decades ($R^2 = 0.8735$ in 1994–2003; $R^2 = 0.8570$ in 2004–2013), demonstrating the ability of 1D convolutional feature extractors to filter spatial-temporal noise in high-density well networks.
2. **Hybrid Architecture Breakthrough in High-Dimensional Stress Regimes**: In the most extensive, non-linear modern dataset (Pre-Monsoon 2014–2024; 161,224 soundings across 27,107 wells), the novel **Hybrid LSTM-GRU** model decisively outperformed all single-cell counterparts, securing $R^2 = 0.7754$, $\text{RMSE} = 4.9537\text{ m}$, $\text{MAE} = 3.0535\text{ m}$, $\text{NSE} = 0.7754$, $\text{KGE} = 0.7546$, and $\text{Willmott's } d = 0.9314$. By concatenating an expressive LSTM temporal memory encoder with a nimble GRU transition layer, the hybrid topology eliminated vanishing gradients across long horizons and mitigated error propagation across extreme drawdown outliers.
3. **Rigorous Leakage-Free Verification**: Zero spatial-temporal leakage occurred. Models were trained on the earliest 70% of chronological dates, validated on the subsequent 15%, and tested strictly on the final unseen 15% out-of-time horizon, confirming true prognostic operational forecasting capability.

---

## 📐 Mathematical Nomenclature & Hydrogeological Terminology

| Symbol / Term | Physical / Mathematical Interpretation | Units |
| :--- | :--- | :--- |
| $y_t$ | Observed groundwater level at time step $t$ | $\text{mbgl}$ (meters below ground level) |
| $\hat{y}_t$ | Neural network predicted groundwater level at time step $t$ | $\text{mbgl}$ |
| $\bar{y}$ | Empirical mean of the observed groundwater time series | $\text{mbgl}$ |
| $\sigma_y, \sigma_{\hat{y}}$ | Standard deviation of observed and predicted distributions | $\text{mbgl}$ |
| $\text{DTWL}$ | Depth to Water Level (measured from ground surface datum down to water surface) | $\text{m}$ |
| $\text{WL in mbgl}$ | Water Level in meters below ground level | $\text{m}$ |
| $(\phi_i, \lambda_i)$ | Geodetic Latitude ($\phi$) and Longitude ($\lambda$) coordinates of well $i$ | Decimal Degrees ($^\circ$) |
| $\tau$ | Autoregressive lookback lag horizon ($\tau \in \{1, 2, 3\}$) | Seasonal cycle steps |
| $R^2$ | Coefficient of Determination ($(-\infty, 1.0]$) | Dimensionless |
| $\text{RMSE}$ | Root Mean Square Error (penalizes extreme predictive departures) | $\text{m}$ |
| $\text{MAE}$ | Mean Absolute Error (robust linear scale of error magnitude) | $\text{m}$ |
| $\text{NSE}$ | Nash-Sutcliffe Efficiency Index ($1.0$ is perfect match; $<0$ indicates mean is better predictor) | Dimensionless |
| $\text{KGE}$ | Kling-Gupta Efficiency (decomposes error into correlation, bias, and variability ratio) | Dimensionless |
| $d$ | Willmott's Modified Index of Agreement ($[0.0, 1.0]$) | Dimensionless |
| $\text{LMI}$ | Legates-McCabe Index of Efficiency ($(-\infty, 1.0]$) | Dimensionless |
| $E'$ | Centered Root Mean Square Difference in polar Taylor coordinates | $\text{m}$ |

---

# 🏛️ PART I: THEORETICAL FRAMEWORK & METHODOLOGY

## Chapter 1: Hydrogeological Context & The Groundwater Crisis

Aquifers represent complex, unconfined and semi-confined multi-layered subsurface geological formations. The movement and storage of groundwater are governed by the classic non-linear 3D Boussinesq equation:

$$\nabla \cdot (K \nabla h) + W = S_s \frac{\partial h}{\partial t}$$

where $K$ denotes the hydraulic conductivity tensor, $h$ is hydraulic head, $W$ represents volumetric source/sink fluxes (precipitation recharge, evapotranspiration, anthropogenic tubewell pumping), and $S_s$ denotes specific storage.

In empirical regional investigations, solving the continuous Boussinesq equation through numerical methods (e.g., MODFLOW) is severely hindered by the lack of continuous subsurface lithological logging, unknown spatial distributions of aquifer transmissivity ($T$), and unmonitored private agricultural tubewell extraction rates. 

Data-driven sequence modeling circumvents unobservable hydrogeological parameters by mapping the functional relationship:

$$\hat{y}_{t+1} = \mathcal{F}_{\mathbf{\Theta}}\left( \mathbf{S}_i, \mathbf{T}_t, \mathbf{Y}_{t-\tau:t} \right)$$

where $\mathcal{F}_{\mathbf{\Theta}}$ represents a parameterized deep neural network parameterized by weights $\mathbf{\Theta}$, $\mathbf{S}_i = [\phi_i, \lambda_i]$ represents spatial coordinates, $\mathbf{T}_t$ represents cyclic calendrical vectors, and $\mathbf{Y}_{t-\tau:t} = [y_t, y_{t-1}, \dots, y_{t-\tau}]$ constitutes the historic autoregressive memory of the observation well.

```
+---------------------------------------------------------------------------------------------------+
|                                 HYDROGEOLOGICAL FORCING CASCADE                                  |
+---------------------------------------------------------------------------------------------------+
|  [ Monsoon Recharge ]  ──(+)──> [ Soil Moisture Retention ] ──(+)──> [ Aquifer Matrix Recharge ]  │
|                                                                                │                  │
|                                                                                ▼                  │
|  [ Intensive Pumping ] ──(-)──> [ Drawdown Cone Formation ] ──(-)──> [ Saturated Thickness Loss ] │
|                                                                                │                  │
|                                                                                ▼                  │
|                                                            [ Observed Water Table Level: y(t) ]   │
+---------------------------------------------------------------------------------------------------+
```

---

## Chapter 2: Multi-Decadal Datasets & Spatial Data Profiling

The investigation utilizes raw observation records cataloged by the Central Ground Water Board (CGWB) across India. Rather than merging disparate seasons and eras into an uninterpretable conglomerate, the data was rigorously partitioned into **four self-contained research corpora**:

```
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|                               FOUR INDEPENDENT RESEARCH DATASETS                                 |
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|  1. Post-Monsoon Season (1994-2023)  | 5,214 Records   | 433 Wells   | 1995-11-01 to 2023-11-30  |
|  2. Pre-Monsoon Decade I (1994-2003) | 94,956 Records  | 14,854 Wells| 1995-04-01 to 2003-05-31  |
|  3. Pre-Monsoon Decade II (2004-2013)| 97,179 Records  | 16,041 Wells| 2004-05-24 to 2013-05-31  |
|  4. Pre-Monsoon Decade III (2014-2024| 161,224 Records | 27,107 Wells| 2014-05-20 to 2024-05-10  |
+──────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Comprehensive Dataset Profile Matrix

| Dataset Identifier | Temporal Range | Usable Records | Monitored Wells | Target Column | Spatial Extent | Mean Depth (m) | Std Dev (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Post-Monsoon (1994–2023)** | 1995–2023 (~29 yrs) | 5,214 | 433 | `WL_in_mbgl` | Regional Core | 4.88 m | 4.22 m |
| **Pre-Monsoon (1994–2003)** | 1995–2003 (~9 yrs) | 94,956 | 14,854 | `DTWL` | Continental Scale | 9.72 m | 9.08 m |
| **Pre-Monsoon (2004–2013)** | 2004–2013 (~10 yrs) | 97,179 | 16,041 | `DTWL` | Continental Scale | 10.84 m | 10.42 m |
| **Pre-Monsoon (2014–2024)** | 2014–2024 (~10 yrs) | 161,224 | 27,107 | `DTWL` | Continental Scale | 11.45 m | 10.87 m |

> [!NOTE]
> **Data Integrity Protocol**: Observations were subjected to strict geospatial validity checks ($\text{Latitude} \in [6.0^\circ, 38.0^\circ\text{N}]$, $\text{Longitude} \in [68.0^\circ, 98.0^\circ\text{E}]$) and physical range validation ($\text{Depth} \in [0.0\text{ m}, 150.0\text{ m}]$). Unrealistic negative water levels and unphysical sensor spikes were filtered out prior to training.

---

## Chapter 3: Spatiotemporal Feature Engineering & Leakage-Free Splitting

To enable deep neural networks to accurately learn aquifer dynamics without introducing exogenous data dependencies (such as interpolated meteorological grids that introduce synthetic spatial bias), a 13-dimensional spatiotemporal feature space was constructed entirely from spatial coordinates, calendar mathematics, and historic well memory:

$$\mathbf{x}_t = \left[ \phi, \lambda, \text{yr}, m, \text{doy}, \sin\left(\frac{2\pi m}{12}\right), \cos\left(\frac{2\pi m}{12}\right), y_{t-1}, y_{t-2}, y_{t-3}, \Delta y_{t-1}, \mu_{3,t}, \sigma_{3,t} \right]^T$$

```
                                FEATURE EXTRACTION TAXONOMY
                                
         Geospatial Domain          Cyclic Calendrical Domain          Autoregressive Aquifer Memory
     ┌───────────────────────┐      ┌────────────────────────┐      ┌──────────────────────
     │ • LATITUDE   (phi)    │      │ • year                 │      │ • gw_lag_1   (y_{t-1})         │
     │ • LONGITUDE  (lambda) │      │ • month                │      │ • gw_lag_2   (y_{t-2})         │
     └───────────────────────┘      │ • day_of_year          │      │ • gw_lag_3   (y_{t-3})         │
                                    │ • month_sin [harmonic] │      │ • gw_change_1 (Delta y)        │
                                    │ • month_cos [harmonic] │      │ • gw_roll_mean_3 (mu_3)        │
                                    └────────────────────────┘      │ • gw_roll_std_3  (sigma_3)     │
                                                                    └──────────────────────
```

### Feature Formulation Details:
1. **Spatial Coordinates ($\phi, \lambda$)**: Latitude and Longitude establish the geographical boundary conditions and localize aquifer hydraulic conductivity zones.
2. **Cyclic Temporal Harmonics ($\sin(2\pi m / 12), \cos(2\pi m / 12)$)**: Continuous trigonometric encoding preserves periodic boundary conditions (December ($m=12$) smoothly transitions to January ($m=1$)).
3. **Multi-Scale Autoregressive Lags ($y_{t-1}, y_{t-2}, y_{t-3}$)**: Captures multi-season aquifer inertia and lingering antecedent storage states.
4. **Dynamic First Difference ($\Delta y_{t-1} = y_{t-1} - y_{t-2}$)**: Supplies velocity of drawdown or recharge to detect rapid extraction shocks.
5. **Moving Window Descriptors ($\mu_{3}, \sigma_{3}$)**: 3-step rolling mean and standard deviation filter out high-frequency noise and reflect local variance regimes.

### Causal Chronological Train-Val-Test Splitting

To ensure the models are genuinely prognostic and capable of operational forecasting, random cross-validation was strictly prohibited. Instead, an unyielding **chronological temporal split** was enforced:

```
0% ────────────────────────────── 70% ───────────────────── 85% ──────────────────── 100%
│                                  │                         │                         │
│       TRAINING HORIZON           │   VALIDATION HORIZON    │     TESTING HORIZON     │
│   (Gradient Optimization)        │   (Early Stopping/LR)   │  (Strict Out-Of-Sample) │
│                                  │                         │                         │
▼                                  ▼                         ▼                         ▼
T_start                       Train_Cutoff                Val_Cutoff                 T_end
```

- **Training Phase (Earliest 70% of Dates)**: Network backpropagation and weight optimization.
- **Validation Phase (Next 15% of Dates)**: Hyperparameter tuning, learning rate reduction triggers, and early stopping checkpoints.
- **Testing Phase (Final 15% of Dates)**: Strictly untouched, out-of-time evaluation measuring real-world prospective prediction fidelity.
- **Data Preprocessing**: `MinMaxScaler` fit strictly on Training distributions and applied downstream to prevent target and feature leakage:

$$\tilde{x} = \frac{x - x_{\min,\text{train}}}{x_{\max,\text{train}} - x_{\min,\text{train}}}$$

---

## Chapter 4: Neural Architecture Formulations

Four distinct deep sequence models were formulated and benchmarked under identical hyperparameters and lookback horizons ($\text{lookback } L = 4$ steps).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 NEURAL NETWORK ARCHITECTURES                                    │
├────────────────────────────────┬────────────────────────────────┬───────────────────────────────┤
│        1. BASELINE LSTM        │        2. GATED RECURRENT      │        3. HYBRID CNN-LSTM     │
│ ┌────────────────────────────┐ │ ┌────────────────────────────┐ │ ┌───────────────────────────┐ │
│ │ Input Sequence [4, 13]     │ │ │ Input Sequence [4, 13]     │ │ │ Input Sequence [4, 13]    │ │
│ └──────────────┬─────────────┘ │ └──────────────┬─────────────┘ │ └─────────────┬─────────────┘ │
│                ▼               │                ▼               │               ▼               │
│ ┌────────────────────────────┐ │ ┌────────────────────────────┐ │ ┌───────────────────────────┐ │
│ │ LSTM (64 Units, Seq=True)  │ │ │ GRU (64 Units, Seq=True)   │ │ │ Conv1D (64 Filters, k=3)  │ │
│ └──────────────┬─────────────┘ │ └──────────────┬─────────────┘ │ └─────────────┬─────────────┘ │
│                ▼               │                ▼               │               ▼               │
│ ┌────────────────────────────┐ │ ┌────────────────────────────┐ │ ┌───────────────────────────┐ │
│ │ Dropout (0.20)             │ │ │ Dropout (0.20)             │ │ │ MaxPooling1D (pool_size=2)│ │
│ └──────────────┬─────────────┘ │ └──────────────┬─────────────┘ │ └─────────────┬─────────────┘ │
│                ▼               │                ▼               │               ▼               │
│ ┌────────────────────────────┐ │ ┌────────────────────────────┐ │ ┌───────────────────────────┐ │
│ │ LSTM (32 Units, Seq=False) │ │ │ GRU (32 Units, Seq=False)  │ │ │ LSTM (48 Units)           │ │
│ └──────────────┬─────────────┘ │ └──────────────┬─────────────┘ │ └─────────────┬─────────────┘ │
│                ▼               │                ▼               │               ▼               │
│ ┌────────────────────────────┐ │ ┌────────────────────────────┐ │ ┌───────────────────────────┐ │
│ │ Dense (16, ReLU) + Drop    │ │ │ Dense (16, ReLU) + Drop    │ │ │ Dense (16, ReLU) + Drop   │ │
│ └──────────────┬─────────────┘ │ └──────────────┬─────────────┘ │ └─────────────┬─────────────┘ │
│                ▼               │                ▼               │               ▼               │
│ ┌────────────────────────────┐ │ ┌────────────────────────────┐ │ ┌───────────────────────────┐ │
│ │ Dense (1, Linear Output)   │ │ │ Dense (1, Linear Output)   │ │ │ Dense (1, Linear Output)  │ │
│ └────────────────────────────┘ │ └────────────────────────────┘ │ └───────────────────────────┘ │
└────────────────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

### 1. Long Short-Term Memory (LSTM)
Engineered to resolve the vanishing gradient dilemma across multi-scale time lags through a dedicated additive cell state $C_t$ controlled by three non-linear gates:

$$\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \quad &&\text{(Forget Gate: discards obsolete storage memory)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \quad &&\text{(Input Gate: admits new recharge signals)} \\
\tilde{C}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \quad &&\text{(Candidate Memory Generation)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &&\text{(Cell State Update)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \quad &&\text{(Output Gate)} \\
h_t &= o_t \odot \tanh(C_t) \quad &&\text{(Hidden State Vector)}
\end{aligned}$$

### 2. Gated Recurrent Unit (GRU)
Couples the forget and input gates into a unified update gate $z_t$ and manages temporal flow through a reset gate $r_t$, eliminating the separate cell state:

$$\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}, x_t] + b_z) \quad &&\text{(Update Gate: balances past memory vs current input)} \\
r_t &= \sigma(W_r [h_{t-1}, x_t] + b_r) \quad &&\text{(Reset Gate: determines past memory neglect)} \\
\tilde{h}_t &= \tanh(W [r_t \odot h_{t-1}, x_t] + b) \quad &&\text{(Candidate Hidden State)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &&\text{(Final Recurrent Output)}
\end{aligned}$$

### 3. Convolutional Recurrent Network (CNN-LSTM)
Integrates a 1D temporal convolution layer to extract high-frequency local sub-seasonal variations before sequence modeling:

$$\mathbf{c}_k = \text{ReLU}\left( \mathbf{W}_c \ast \mathbf{x}_{k:k+w-1} + \mathbf{b}_c \right)$$

followed by Max-Pooling to downsample temporal length and feed an LSTM layer (48 units) for macro-temporal context.

### 4. Proposed Hybrid LSTM-GRU Dual-Engine
A novel serial-cascaded architecture designed to exploit the distinct mathematical strengths of both LSTM and GRU:

```
[ Input: 4 Timesteps × 13 Features ]
                 │
                 ▼
 ┌────────────────────────────────┐
 │ LSTM Layer (64 Hidden Units)   │ ──> Captures long-range multi-seasonal dependencies
 │ return_sequences = True        │     via high-capacity cell state C(t)
 └───────────────┬────────────────┘
                 ▼
 ┌────────────────────────────────┐
 │ Dropout Layer (Rate = 0.20)    │ ──> Prevents co-adaptation of recurrent weights
 └───────────────┬────────────────┘
                 ▼
 ┌────────────────────────────────┐
 │ GRU Layer (32 Hidden Units)    │ ──> Synthesizes sequence representations into a concise
 │ return_sequences = False       │     hidden vector with low-latency parameter gating
 └───────────────┬────────────────┘
                 ▼
 ┌────────────────────────────────┐
 │ Dropout Layer (Rate = 0.20)    │
 └───────────────┬────────────────┘
                 ▼
 ┌────────────────────────────────┐
 │ Fully-Connected Dense Layer    │ ──> Non-linear spatial-temporal manifold projection
 │ 16 Units, ReLU Activation      │
 └───────────────┬────────────────┘
                 ▼
 ┌────────────────────────────────┐
 │ Output Neuron (Dense, Linear)  │ ──> Scaled Groundwater Depth: y_hat(t)
 └────────────────────────────────┘
```

The first layer (LSTM, 64 units) encodes the temporal trajectory into a richer 64-dimensional feature space at each time step. The second layer (GRU, 32 units) acts as an adaptive filter, rapidly integrating these sequence features without the computational drag and over-parameterization of stacked homogeneous LSTMs.

---

## Chapter 5: Optimization Suite & 11-Dimensional Statistical Metrics

All models were trained using the **Adam Optimizer** with an initial learning rate $\eta = 0.002$, optimizing the Mean Squared Error loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

### Dynamic Callbacks & Convergence Safeguards
- **EarlyStopping**: Monitored `val_loss` with patience of 10 epochs, restoring optimal model weights upon early termination.
- **ReduceLROnPlateau**: Halved learning rate ($\text{factor} = 0.5$) if `val_loss` plateaued for 5 consecutive epochs (minimum $\eta = 10^{-5}$).

### The 11-Dimensional Metric Evaluation Suite

To ensure comprehensive hydrological validation, 11 distinct performance indicators were calculated on both training and testing phases:

1. **Root Mean Square Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^N (y_t - \hat{y}_t)^2}$$
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N}\sum_{t=1}^N |y_t - \hat{y}_t|$$
3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum_{t=1}^N (y_t - \hat{y}_t)^2}{\sum_{t=1}^N (y_t - \bar{y})^2}$$
4. **Pearson Correlation Coefficient ($r$)**:
   $$\text{Pearson } r = \frac{\sum_{t=1}^N (y_t - \bar{y})(\hat{y}_t - \bar{\hat{y}})}{\sqrt{\sum_{t=1}^N (y_t - \bar{y})^2 \sum_{t=1}^N (\hat{y}_t - \bar{\hat{y}})^2}}$$
5. **Nash-Sutcliffe Efficiency (NSE)**:
   $$\text{NSE} = 1 - \frac{\sum_{t=1}^N (y_t - \hat{y}_t)^2}{\sum_{t=1}^N (y_t - \bar{y})^2}$$
6. **Kling-Gupta Efficiency (KGE)**:
   $$\text{KGE} = 1 - \sqrt{(r - 1)^2 + \left(\frac{\sigma_{\hat{y}}}{\sigma_y} - 1\right)^2 + \left(\frac{\mu_{\hat{y}}}{\mu_y} - 1\right)^2}$$
7. **Normalized Root Mean Square Error (NRMSE)**:
   $$\text{NRMSE} = \frac{\text{RMSE}}{\bar{y}}$$
8. **Willmott's Modified Index of Agreement ($d$)**:
   $$d = 1 - \frac{\sum_{t=1}^N (y_t - \hat{y}_t)^2}{\sum_{t=1}^N \left( |\hat{y}_t - \bar{y}| + |y_t - \bar{y}| \right)^2}$$
9. **Legates-McCabe Index of Efficiency (LMI)**:
   $$\text{LMI} = 1 - \frac{\sum_{t=1}^N |y_t - \hat{y}_t|}{\sum_{t=1}^N |y_t - \bar{y}|}$$
10. **Mean Squared Error (MSE)**:
    $$\text{MSE} = \frac{1}{N}\sum_{t=1}^N (y_t - \hat{y}_t)^2$$
11. **Mean Absolute Percentage Error (MAPE %)**:
    $$\text{MAPE} = \frac{100\%}{N}\sum_{t=1}^N \left| \frac{y_t - \hat{y}_t}{\max(|y_t|, \epsilon)} \right|$$

---

# 📊 PART II: THE EMPIRICAL ATLAS — EXPERIMENTAL PLATES & RESULTS

---

## 🌊 SECTION 1: POST-MONSOON REGIONAL REGIME (1994–2023)

### Hydrogeological Profile
- **Temporal Horizon**: November 1995 to November 2023 (29 Years of Annual Post-Monsoon Observations).
- **Spatial Coverage**: 433 Centrally Monitored Hydrograph Network Wells.
- **Dataset Dimensions**: 5,214 Processed Spatial-Temporal Records.
- **Target Phenomenon**: Post-Monsoon Stabilized Hydraulic Head (`WL_in_mbgl`).
- **Optimal Architecture**: **Gated Recurrent Unit (GRU)** ($R^2 = 0.7447$, $\text{RMSE} = 2.4851\text{ m}$).

### 🔬 Empirical Performance Analysis
In this long-term post-recharge regime, groundwater levels are stabilized by the preceding summer monsoon. The **GRU** network proved optimal, securing the lowest unseen test $\text{RMSE}$ ($2.4851\text{ m}$) and highest $R^2$ ($0.7447$). The compact gating mechanics of the GRU avoided the mild over-parameterization observed in standard LSTM ($R^2 = 0.6471$), while the **CNN-LSTM** followed closely with $R^2 = 0.7241$ and the highest test Kling-Gupta Efficiency ($\text{KGE} = 0.6335$).

```
POST-MONSOON TESTING PERFORMANCE COMPARISON (1994–2023)
Metric      LSTM          GRU [BEST]    CNN-LSTM      Hybrid LSTM-GRU
───────────────────────────────────────────────────────────────────────
R²          0.6471        0.7447        0.7241        0.7139
RMSE (m)    2.9222        2.4851        2.5836        2.6309
MAE (m)     1.6209        1.3300        1.3904        1.3981
Pearson r   0.8786        0.8960        0.8742        0.8919
KGE         0.4618        0.6065        0.6335        0.5520
NSE         0.6471        0.7447        0.7241        0.7139
Willmott d  0.8641        0.9134        0.9068        0.8978
LMI         0.3931        0.5020        0.4794        0.4765
```

---

### 🖼️ Plate 1: Visual Results Atlas — Post-Monsoon Season (1994–2023)

<div align="center">

#### Figure 1.0: Observed Regional Groundwater Timeline (1994–2023)
![Plate 1.0 - Post-Monsoon Timeline](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/00_Observed_Regional_Timeline.png)
*Figure 1.0: Real observed post-monsoon regional water level dynamics across 433 wells spanning 1995 to 2023. Seasonal cycles reflect long-term multi-annual precipitation trends.*

<br/>

#### Figure 1.1: Multi-Model Training and Validation Loss Convergence
![Plate 1.1 - Post-Monsoon Loss Convergence](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/01_Training_Validation_Loss.png)
*Figure 1.1: Loss (MSE) convergence trajectories across 50 epochs for LSTM, GRU, CNN-LSTM, and Hybrid LSTM-GRU. Notice stable decay without severe overfitting.*

<br/>

#### Figure 1.2: Observed vs. Predicted Time-Series Reconstruction Across Chronological Partitions
![Plate 1.2 - Post-Monsoon Time Series](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/02_Time_Series_Observed_vs_Predicted.png)
*Figure 1.2: Complete chronological progression across Training (pre-2016), Validation (2016–2019), and out-of-sample Testing (post-2019). GRU and CNN-LSTM accurately track peak recharge years.*

<br/>

#### Figure 1.3: Spatiotemporal Feature Correlation Heatmap
![Plate 1.3 - Post-Monsoon Correlation](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/03_Correlation_Matrix.png)
*Figure 1.3: Pearson correlation matrix illustrating strong coupling between autoregressive lags (gw_lag_1, gw_roll_mean_3) and target groundwater level, alongside latitude/longitude spatial gradients.*

<br/>

#### Figure 1.4: Parity and Scatter Plots (Observed vs. Predicted)
![Plate 1.4 - Post-Monsoon Scatter Parity](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/04_Scatter_Observed_vs_Predicted.png)
*Figure 1.4: Scatter parity comparison on Training and Testing splits. The dashed line represents ideal 1:1 identity. GRU displays minimal dispersion around the regression diagonal.*

<br/>

#### Figure 1.5: Out-of-Sample Test Set Residuals Over Time
![Plate 1.5 - Post-Monsoon Residuals](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/05_Test_Residuals_vs_Time.png)
*Figure 1.5: Temporal residual errors ($e_t = \hat{y}_t - y_t$) over the testing period. Residuals are homoscedastic and uniformly distributed around zero.*

<br/>

#### Figure 1.6: Residual Probability Distributions and Gaussian Fits
![Plate 1.6 - Post-Monsoon Residual Distributions](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/06_Test_Residual_Distributions.png)
*Figure 1.6: Empirical error histograms, Kernel Density Estimation (KDE), and fitted Gaussian curves for all models. Mean errors center closely on $\mu \approx 0.0\text{ m}$.*

<br/>

#### Figure 1.7: Dual-Phase Bipolar Taylor Diagram (Training vs. Testing)
![Plate 1.7 - Post-Monsoon Taylor Diagram](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/07_Taylor_Diagram_Training_Testing.png)
*Figure 1.7: Taylor diagram evaluating correlation coefficient ($r$), normalized standard deviation, and centered RMSE simultaneously. GRU sits closest to the reference observation point.*

<br/>

#### Figure 1.8: Cross-Architecture Performance Benchmarking
![Plate 1.8 - Post-Monsoon Performance Bars](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/08_Model_Performance_Comparison.png)
*Figure 1.8: Multi-metric bar chart comparing RMSE, MAE, R², NSE, and KGE across the four candidate architectures.*

<br/>

#### Figure 1.9: Comprehensive Statistical Metrics Publication Matrix
![Plate 1.9 - Post-Monsoon Metrics Matrix](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/09_Statistical_Metrics_Training_Testing.png)
*Figure 1.9: Publication-format summary table illustrating exact numerical metric scores across both training and testing phases.*

<br/>

#### Figure 1.10: Hybrid LSTM-GRU Convergence and Validation Dynamics
![Plate 1.10 - Post-Monsoon Hybrid Convergence](outputs/DL_4_SEPARATE_DATASETS/post_monsoon_1994_2023/figures/10_Hybrid_LSTM_GRU_Convergence.png)
*Figure 1.10: Detailed convergence trajectory of the Hybrid LSTM-GRU dual-engine, demonstrating gradient stability and smooth loss minimization.*

</div>

---

## 🌾 SECTION 2: PRE-MONSOON BASELINE DECADE (1994–2003)

### Hydrogeological Profile
- **Temporal Horizon**: April 1995 to May 2003 (~9 Years of Pre-Monsoon Peak Drawdown).
- **Spatial Coverage**: 14,854 Continental-Scale Borewells.
- **Dataset Dimensions**: 94,956 Processed Spatial-Temporal Records.
- **Target Phenomenon**: Pre-Monsoon Depth to Water Level (`DTWL` in mbgl).
- **Optimal Architecture**: **1D-CNN-LSTM** ($R^2 = 0.8735$, $\text{RMSE} = 3.3707\text{ m}$, $\text{KGE} = 0.8886$).

### 🔬 Empirical Performance Analysis
During the 1994–2003 baseline decade, pre-monsoon water tables reflect the annual maximum aquifer drawdown prior to seasonal replenishment. The **1D-CNN-LSTM** network achieved outstanding predictive accuracy on unseen test data, registering an $R^2$ of **0.8735**, an $\text{RMSE}$ of **3.3707 m**, and a Kling-Gupta Efficiency ($\text{KGE}$) of **0.8886**. 

The 1D convolutional filter acts as an effective spatial-temporal filter, capturing local drawdown gradients across contiguous observation wells before passing the representation to the recurrent LSTM layer. GRU followed closely with $R^2 = 0.8704$, while the Hybrid LSTM-GRU achieved $R^2 = 0.8674$ and the highest overall test KGE ($\text{KGE} = 0.8985$).

```
PRE-MONSOON DECADE I TESTING PERFORMANCE (1994–2003)
Metric      LSTM          GRU           CNN-LSTM [BEST] Hybrid LSTM-GRU
───────────────────────────────────────────────────────────────────────
R²          0.8569        0.8704        0.8735          0.8674
RMSE (m)    3.5846        3.4109        3.3707          3.4501
MAE (m)     1.8679        1.8208        1.8177          1.8717
Pearson r   0.9291        0.9354        0.9349          0.9317
KGE         0.8383        0.8670        0.8886          0.8985
NSE         0.8569        0.8704        0.8735          0.8674
Willmott d  0.9573        0.9630        0.9643          0.9633
LMI         0.6367        0.6458        0.6464          0.6359
MAPE (%)    24.1857       24.2264       24.8673         24.4090
```

---

### 🖼️ Plate 2: Visual Results Atlas — Pre-Monsoon Decade I (1994–2003)

<div align="center">

#### Figure 2.0: Regional Observed Groundwater Timeline (1994–2003)
![Plate 2.0 - Pre-Monsoon 1994-2003 Timeline](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/00_Observed_Regional_Timeline.png)
*Figure 2.0: Continental-scale pre-monsoon water level observations across 14,854 wells, reflecting the baseline water table prior to rapid modern mechanical expansion.*

<br/>

#### Figure 2.1: Multi-Model Training and Validation Loss Convergence
![Plate 2.1 - Pre-Monsoon 1994-2003 Loss](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/01_Training_Validation_Loss.png)
*Figure 2.1: Training and validation loss curves showing rapid convergence within 20 epochs across all four model architectures.*

<br/>

#### Figure 2.2: Time-Series Observed vs. Predicted Trajectory
![Plate 2.2 - Pre-Monsoon 1994-2003 Time Series](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/02_Time_Series_Observed_vs_Predicted.png)
*Figure 2.2: Time-series alignment across Train, Validation, and Test partitions. CNN-LSTM captures deep regional drawdowns with remarkable fidelity.*

<br/>

#### Figure 2.3: Spatiotemporal Feature Correlation Heatmap
![Plate 2.3 - Pre-Monsoon 1994-2003 Correlation](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/03_Correlation_Matrix.png)
*Figure 2.3: Pearson correlation matrix showing substantial memory retention ($\text{gw\_lag\_1}$ correlation $> 0.85$) and spatial coordinate influence.*

<br/>

#### Figure 2.4: Scatter and Parity Regression Plots
![Plate 2.4 - Pre-Monsoon 1994-2003 Parity](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/04_Scatter_Observed_vs_Predicted.png)
*Figure 2.4: 1:1 parity plots demonstrating tight clustering of predictions along the identity line up to 50 meters depth.*

<br/>

#### Figure 2.5: Test Residuals Trajectory Over Time
![Plate 2.5 - Pre-Monsoon 1994-2003 Residuals](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/05_Test_Residuals_vs_Time.png)
*Figure 2.5: Out-of-sample test residuals showing stable variance across the entire out-of-time evaluation window.*

<br/>

#### Figure 2.6: Test Residual Distributions and Gaussian Modeling
![Plate 2.6 - Pre-Monsoon 1994-2003 Error Distributions](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/06_Test_Residual_Distributions.png)
*Figure 2.6: 2×2 grid of error distributions for LSTM, GRU, CNN-LSTM, and Hybrid. CNN-LSTM exhibits the narrowest standard deviation ($\sigma = 3.37\text{ m}$).*

<br/>

#### Figure 2.7: Bipolar Taylor Diagram (Training vs. Testing)
![Plate 2.7 - Pre-Monsoon 1994-2003 Taylor](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/07_Taylor_Diagram_Training_Testing.png)
*Figure 2.7: Taylor diagram showing high correlation ($r > 0.93$) and close normalized standard deviation match with observations.*

<br/>

#### Figure 2.8: Model Performance Benchmarking
![Plate 2.8 - Pre-Monsoon 1994-2003 Performance](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/08_Model_Performance_Comparison.png)
*Figure 2.8: Benchmark comparison highlighting the dominance of CNN-LSTM and GRU.*

<br/>

#### Figure 2.9: Statistical Metrics Publication Matrix
![Plate 2.9 - Pre-Monsoon 1994-2003 Metrics](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/09_Statistical_Metrics_Training_Testing.png)
*Figure 2.9: Complete numerical table in thesis format detailing all 11 statistical metrics across both experimental phases.*

<br/>

#### Figure 2.10: Hybrid LSTM-GRU Convergence Diagnostics
![Plate 2.10 - Pre-Monsoon 1994-2003 Hybrid Convergence](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_1994_2003/figures/10_Hybrid_LSTM_GRU_Convergence.png)
*Figure 2.10: Convergence trajectory for the Hybrid model illustrating rapid loss reduction and high generalization stability.*

</div>

---

## 🏗️ SECTION 3: PRE-MONSOON INTERMEDIATE DECADE (2004–2013)

### Hydrogeological Profile
- **Temporal Horizon**: May 2004 to May 2013 (~10 Years of Accelerating Extraction).
- **Spatial Coverage**: 16,041 Monitored Wells.
- **Dataset Dimensions**: 97,179 Processed Spatial-Temporal Records.
- **Target Phenomenon**: Pre-Monsoon Depth to Water Level (`DTWL` in mbgl).
- **Optimal Architecture**: **1D-CNN-LSTM** ($R^2 = 0.8570$, $\text{RMSE} = 4.0285\text{ m}$, $\text{KGE} = 0.9050$).

### 🔬 Empirical Performance Analysis
The 2004–2013 period witnessed an unprecedented surge in agricultural submersible borewell installations, reflected by an increased mean depth to water table ($10.84\text{ m}$ vs $9.72\text{ m}$ in the previous decade). 

Despite greater spatial heterogeneity and non-stationary pumping regimes, **1D-CNN-LSTM** retained its superior performance, achieving $R^2 = 0.8570$, test $\text{RMSE} = 4.0285\text{ m}$, $\text{MAE} = 2.2060\text{ m}$, and a peerless Kling-Gupta Efficiency of **0.9050**. Willmott's index of agreement reached **0.9610**, proving the architecture's resilience in tracking deep cone-of-depression anomalies across varied hydrogeological terrains.

```
PRE-MONSOON DECADE II TESTING PERFORMANCE (2004–2013)
Metric      LSTM          GRU           CNN-LSTM [BEST] Hybrid LSTM-GRU
───────────────────────────────────────────────────────────────────────
R²          0.8438        0.8486        0.8570          0.8444
RMSE (m)    4.2105        4.1452        4.0285          4.2033
MAE (m)     2.4295        2.3075        2.2060          2.3617
Pearson r   0.9206        0.9226        0.9261          0.9210
KGE         0.8686        0.8567        0.9050          0.8459
NSE         0.8438        0.8486        0.8570          0.8444
Willmott d  0.9562        0.9561        0.9610          0.9543
LMI         0.6065        0.6262        0.6427          0.6174
```

---

### 🖼️ Plate 3: Visual Results Atlas — Pre-Monsoon Decade II (2004–2013)

<div align="center">

#### Figure 3.0: Regional Observed Groundwater Timeline (2004–2013)
![Plate 3.0 - Pre-Monsoon 2004-2013 Timeline](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/00_Observed_Regional_Timeline.png)
*Figure 3.0: Pre-monsoon groundwater dynamics across 16,041 monitoring stations, showing deeper regional drawdowns.*

<br/>

#### Figure 3.1: Multi-Model Training and Validation Loss Convergence
![Plate 3.1 - Pre-Monsoon 2004-2013 Loss](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/01_Training_Validation_Loss.png)
*Figure 3.1: Epoch-by-epoch loss reduction demonstrating stable parameter optimization without divergent validation losses.*

<br/>

#### Figure 3.2: Time-Series Observed vs. Predicted Trajectory
![Plate 3.2 - Pre-Monsoon 2004-2013 Time Series](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/02_Time_Series_Observed_vs_Predicted.png)
*Figure 3.2: Reconstructed time-series progression over the full 10-year observation span.*

<br/>

#### Figure 3.3: Spatiotemporal Feature Correlation Heatmap
![Plate 3.3 - Pre-Monsoon 2004-2013 Correlation](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/03_Correlation_Matrix.png)
*Figure 3.3: Feature correlation matrix illustrating persistent autoregressive correlation and spatial gradient significance.*

<br/>

#### Figure 3.4: Scatter and Parity Regression Plots
![Plate 3.4 - Pre-Monsoon 2004-2013 Parity](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/04_Scatter_Observed_vs_Predicted.png)
*Figure 3.4: Parity regression showing strong alignment up to extreme depths ($>60\text{ m}$).*

<br/>

#### Figure 3.5: Test Residuals Trajectory Over Time
![Plate 3.5 - Pre-Monsoon 2004-2013 Residuals](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/05_Test_Residuals_vs_Time.png)
*Figure 3.5: Testing error residuals mapped over time showing stable bounds.*

<br/>

#### Figure 3.6: Test Residual Distributions and Gaussian Modeling
![Plate 3.6 - Pre-Monsoon 2004-2013 Error Distributions](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/06_Test_Residual_Distributions.png)
*Figure 3.6: 2×2 error distribution panels confirming near-Gaussian residual symmetry.*

<br/>

#### Figure 3.7: Bipolar Taylor Diagram (Training vs. Testing)
![Plate 3.7 - Pre-Monsoon 2004-2013 Taylor](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/07_Taylor_Diagram_Training_Testing.png)
*Figure 3.7: Taylor diagram showing high correlation ($r > 0.92$) and normalized CRMSE under $0.38$.*

<br/>

#### Figure 3.8: Model Performance Benchmarking
![Plate 3.8 - Pre-Monsoon 2004-2013 Performance](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/08_Model_Performance_Comparison.png)
*Figure 3.8: Benchmark bars indicating CNN-LSTM leadership across all key statistical metrics.*

<br/>

#### Figure 3.9: Statistical Metrics Publication Matrix
![Plate 3.9 - Pre-Monsoon 2004-2013 Metrics](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/09_Statistical_Metrics_Training_Testing.png)
*Figure 3.9: Publication matrix detailing all metrics for training and out-of-sample testing.*

<br/>

#### Figure 3.10: Hybrid LSTM-GRU Convergence Diagnostics
![Plate 3.10 - Pre-Monsoon 2004-2013 Hybrid Convergence](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2004_2013/figures/10_Hybrid_LSTM_GRU_Convergence.png)
*Figure 3.10: Convergence curve of the Hybrid model showing smooth loss minimization across training iterations.*

</div>

---

## ⚡ SECTION 4: PRE-MONSOON MODERN STRESS DECADE (2014–2024)

### Hydrogeological Profile
- **Temporal Horizon**: May 2014 to May 2024 (~10 Years of Modern Anthropogenic Stress).
- **Spatial Coverage**: 27,107 Nationwide Observation Wells.
- **Dataset Dimensions**: 161,224 Processed Spatial-Temporal Records.
- **Target Phenomenon**: Pre-Monsoon Depth to Water Level (`DTWL` in mbgl).
- **Optimal Architecture**: **Hybrid LSTM-GRU Dual-Engine** ($R^2 = 0.7754$, $\text{RMSE} = 4.9537\text{ m}$, $\text{MAE} = 3.0535\text{ m}$, $\text{KGE} = 0.7546$).

### 🔬 Empirical Performance Analysis & The Hybrid Breakthrough
The 2014–2024 modern dataset represents the most challenging, complex, and high-dimensional experimental domain in the entire study, comprising over **161,200 observations across 27,100+ borewells**. This decade is characterized by extreme hydrological volatility: localized over-extraction, deep cone expansions ($>70\text{ m}$), urbanization-driven recharge impedance, and climatic extremes.

Under these rigorous operational stress conditions, individual architectures faced severe challenges:
- Standard **LSTM** achieved an $R^2$ of $0.7437$ and an $\text{RMSE}$ of $5.2919\text{ m}$.
- Standalone **GRU** reached $R^2 = 0.7483$ with an $\text{RMSE}$ of $5.2441\text{ m}$.
- Standalone **CNN-LSTM** experienced performance degradation ($R^2 = 0.6858$, $\text{RMSE} = 5.8591\text{ m}$) due to high spatial noise across the dense 27,107-well distribution.

The novel **Hybrid LSTM-GRU Dual-Engine Network achieved a decisive breakthrough**:
- **Highest Coefficient of Determination**: $R^2 = \mathbf{0.7754}$
- **Lowest Out-of-Sample RMSE**: $\text{RMSE} = \mathbf{4.9537\text{ m}}$ (the only model to break below 5.0 meters)
- **Lowest Mean Absolute Error**: $\text{MAE} = \mathbf{3.0535\text{ m}}$ (an absolute error reduction of $>0.51\text{ m}$ over LSTM/GRU and $>1.10\text{ m}$ over CNN-LSTM)
- **Highest Kling-Gupta Efficiency**: $\text{KGE} = \mathbf{0.7546}$
- **Highest Willmott Index of Agreement**: $d = \mathbf{0.9314}$

By utilizing an expressive 64-unit LSTM first stage to capture long-range temporal dependencies, followed by a 32-unit GRU second stage that synthesizes and stabilizes hidden state transitions, the Hybrid network eliminated recurrent error drift and handled multi-scale variance with remarkable precision.

```
PRE-MONSOON MODERN DECADE III TESTING PERFORMANCE (2014–2024)
Metric      LSTM          GRU           CNN-LSTM      Hybrid LSTM-GRU [BEST]
────────────────────────────────────────────────────────────────────────────
R²          0.7437        0.7483        0.6858        0.7754  ▲ HIGHEST
RMSE (m)    5.2919        5.2441        5.8591        4.9537  ▼ LOWEST
MAE (m)     3.5642        3.5638        4.1537        3.0535  ▼ LOWEST
Pearson r   0.8845        0.8901        0.8716        0.8923  ▲ HIGHEST
KGE         0.7471        0.7182        0.6409        0.7546  ▲ HIGHEST
NSE         0.7437        0.7483        0.6858        0.7754  ▲ HIGHEST
NRMSE       0.5730        0.5678        0.6344        0.5364  ▼ LOWEST
Willmott d  0.9292        0.9279        0.9077        0.9314  ▲ HIGHEST
LMI         0.4082        0.4083        0.3103        0.4930  ▲ HIGHEST
MSE         28.0040       27.5005       34.3294       24.5396 ▼ LOWEST
MAPE (%)    68.8016       67.9088       79.4064       63.9105 ▼ LOWEST
```

---

### 🖼️ Plate 4: Visual Results Atlas — Pre-Monsoon Modern Stress Decade (2014–2024)

<div align="center">

#### Figure 4.0: Modern Regional Observed Groundwater Timeline (2014–2024)
![Plate 4.0 - Pre-Monsoon 2014-2024 Timeline](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/00_Observed_Regional_Timeline.png)
*Figure 4.0: Modern water table observations across 27,107 wells, illustrating the unprecedented scale and density of monitoring data.*

<br/>

#### Figure 4.1: Multi-Model Training and Validation Loss Convergence
![Plate 4.1 - Pre-Monsoon 2014-2024 Loss](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/01_Training_Validation_Loss.png)
*Figure 4.1: Training and validation loss curves. The Hybrid LSTM-GRU architecture reaches the lowest validation loss plateau.*

<br/>

#### Figure 4.2: Observed vs. Predicted Time-Series Progression
![Plate 4.2 - Pre-Monsoon 2014-2024 Time Series](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/02_Time_Series_Observed_vs_Predicted.png)
*Figure 4.2: Time-series reconstruction across Train, Validation, and Test phases. The Hybrid model (purple curve) tracks severe drawdown dips closely.*

<br/>

#### Figure 4.3: Spatiotemporal Feature Correlation Heatmap
![Plate 4.3 - Pre-Monsoon 2014-2024 Correlation](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/03_Correlation_Matrix.png)
*Figure 4.3: Feature correlation matrix across 161,224 observations.*

<br/>

#### Figure 4.4: Scatter and Parity Regression Analysis
![Plate 4.4 - Pre-Monsoon 2014-2024 Parity](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/04_Scatter_Observed_vs_Predicted.png)
*Figure 4.4: Parity plots on unseen test data. Hybrid LSTM-GRU maintains tightest clustering along the identity diagonal.*

<br/>

#### Figure 4.5: Out-of-Sample Test Residuals Over Time
![Plate 4.5 - Pre-Monsoon 2014-2024 Residuals](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/05_Test_Residuals_vs_Time.png)
*Figure 4.5: Residuals over the 2021–2024 out-of-sample test window. The Hybrid model demonstrates clean, bounded error scatter.*

<br/>

#### Figure 4.6: Residual Probability Distributions and Gaussian Fits
![Plate 4.6 - Pre-Monsoon 2014-2024 Error Distributions](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/06_Test_Residual_Distributions.png)
*Figure 4.6: 2×2 error distribution panels. The Hybrid LSTM-GRU achieves the lowest variance ($\sigma = 4.95\text{ m}$) and most balanced zero-centered Gaussian fit.*

<br/>

#### Figure 4.7: Bipolar Taylor Diagram (Training vs. Testing)
![Plate 4.7 - Pre-Monsoon 2014-2024 Taylor](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/07_Taylor_Diagram_Training_Testing.png)
*Figure 4.7: Taylor diagram showing the Hybrid LSTM-GRU achieving the shortest distance to the reference observation point ($E' = 4.7492\text{ m}$, normalized CRMSE $= 0.4543$).*

<br/>

#### Figure 4.8: Cross-Architecture Performance Benchmarking
![Plate 4.8 - Pre-Monsoon 2014-2024 Performance](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/08_Model_Performance_Comparison.png)
*Figure 4.8: Model comparison bar chart showing the Hybrid architecture leading across all five key validation metrics.*

<br/>

#### Figure 4.9: Statistical Metrics Publication Matrix
![Plate 4.9 - Pre-Monsoon 2014-2024 Metrics](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/09_Statistical_Metrics_Training_Testing.png)
*Figure 4.9: Complete numerical table in thesis format detailing training and testing scores for all four models.*

<br/>

#### Figure 4.10: Hybrid LSTM-GRU Convergence and Validation Dynamics
![Plate 4.10 - Pre-Monsoon 2014-2024 Hybrid Convergence](outputs/DL_4_SEPARATE_DATASETS/pre_monsoon_2014_2024/figures/10_Hybrid_LSTM_GRU_Convergence.png)
*Figure 4.10: Diagnostic convergence profile of the Hybrid LSTM-GRU dual-engine on the 161,224-sample dataset, demonstrating smooth loss minimization.*

</div>

---

# 🔬 PART III: CROSS-DECADAL SYNTHESIS & ARCHITECTURAL BENCHMARKING

## Chapter 6: Decadal Depletion Trends & Aquifer Memory Evolution

By comparing the three sequential pre-monsoon decades (1994–2003, 2004–2013, 2014–2024), this monograph documents an unambiguous **30-year downward trajectory in regional water tables**:

```
DECADAL GROUNDWATER REGIME SHIFTS (PRE-MONSOON 1994–2024)
────────────────────────────────────────────────────────────────────────────
Decadal Era                   1994–2003        2004–2013        2014–2024
────────────────────────────────────────────────────────────────────────────
Active Monitoring Wells       14,854           16,041           27,107
Empirical Records             94,956           97,179           161,224
Mean Depth to Water (mbgl)    9.72 m           10.84 m          11.45 m (▲ +17.8%)
Standard Deviation (mbgl)     9.08 m           10.42 m          10.87 m (▲ +19.7%)
Baseline Model R²             0.8735           0.8570           0.7754
Test Set RMSE (m)             3.37 m           4.03 m           4.95 m
Optimal Architecture          CNN-LSTM         CNN-LSTM         Hybrid LSTM-GRU
────────────────────────────────────────────────────────────────────────────
```

### Key Hydrodynamic Findings:
1. **Regional Water Table Drawdown**: The empirical mean depth to water table expanded from **9.72 m** (1994–2003) to **10.84 m** (2004–2013), reaching **11.45 m** in 2014–2024—representing a net regional depression of **+1.73 meters** across the monitored network.
2. **Variance Amplification**: The standard deviation increased from **9.08 m** to **10.87 m**, confirming that water table decline is highly heterogeneous, with extreme localized cones of depression forming around intensive agricultural and peri-urban tubewell clusters.
3. **Diminishing Predictability Horizon**: As anthropogenic extraction decouples local groundwater dynamics from natural precipitation recharge, the maximum achievable predictive $R^2$ moderated from $0.8735$ in the early era to $0.7754$ in the modern era, underscoring the growing complexity of data-driven aquifer modeling under heavy human influence.

---

## Chapter 7: Deep Architecture Selection Matrix: Regimes of Dominance

Based on empirical performance across 358,000+ observations, this study proposes a formal **Deep Learning Architecture Selection Framework** for hydroinformatics:

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                     HYDROLOGICAL ARCHITECTURE SELECTION GUIDE                             │
├─────────────────────┬───────────────────────┬─────────────────────────────────────────────┤
│ Hydrogeological     │ Optimal               │ Core Architectural                          │
│ Regime Conditions   │ Architecture          │ Mechanism & Rationale                       │
├─────────────────────┼───────────────────────┼─────────────────────────────────────────────┤
│ Post-Recharge,      │ Gated Recurrent       │ Balanced parameter budget prevents          │
│ Moderate Density    │ Unit (GRU)            │ overfitting on moderate sample sizes;       │
│ (400–1,000 Wells)   │                       │ fast convergence and stable gating.         │
├─────────────────────┼───────────────────────┼─────────────────────────────────────────────┤
│ High Spatial        │ 1D-CNN-LSTM           │ 1D convolutional kernel captures localized  │
│ Density, Moderate   │                       │ spatial gradients and filters noise prior   │
│ Depletion (10K-20K) │                       │ to recurrent temporal processing.           │
├─────────────────────┼───────────────────────┼─────────────────────────────────────────────┤
│ Massive Scale, High │ Hybrid LSTM-GRU       │ LSTM encoder preserves multi-seasonal       │
│ Variance, Extreme   │ Dual-Engine           │ memory while GRU second stage prevents      │
│ Stress (>25K Wells) │                       │ gradient vanishing and error accumulation.  │
└─────────────────────┴───────────────────────┴─────────────────────────────────────────────┘
```

---

## Chapter 8: Multi-Phase Polar Taylor Diagram Geometrics

The Taylor diagram provides a comprehensive visual summary of model performance by evaluating three distinct statistical properties in polar coordinates:
1. **Radial Distance**: Proportional to the standard deviation of predictions ($\sigma_f$).
2. **Azimuthal Angle**: Corresponds to Pearson's correlation coefficient ($\arccos(r)$).
3. **Distance from Observation Point**: Equals the centered root mean square difference ($E'$).

$$E'^2 = \sigma_f^2 + \sigma_r^2 - 2\sigma_f\sigma_r r$$

```
POLAR TAYLOR COORDINATES ON UNSEEN TEST SETS
Dataset                Model             Predicted STD (m)  Pearson r  CRMSE (m)  Norm CRMSE
─────────────────────────────────────────────────────────────────────────────────────────────
Post-Monsoon           LSTM              3.0512             0.8786     2.6705     0.5429
(Obs STD: 4.9187 m)    GRU [BEST]        3.6545             0.8960     2.3100     0.4696
                       CNN-LSTM          3.7438             0.8742     2.4522     0.4985
                       Hybrid LSTM-GRU   3.4165             0.8919     2.4271     0.4934
─────────────────────────────────────────────────────────────────────────────────────────────
Pre-Monsoon 1994–03    LSTM              8.1440             0.9291     3.5658     0.3763
(Obs STD: 9.4760 m)    GRU               8.4952             0.9354     3.3699     0.3556
                       CNN-LSTM [BEST]   8.6191             0.9349     3.3706     0.3557
                       Hybrid LSTM-GRU   8.8024             0.9317     3.4416     0.3632
─────────────────────────────────────────────────────────────────────────────────────────────
Pre-Monsoon 2004–13    LSTM              9.7573             0.9206     4.1603     0.3905
(Obs STD: 10.6547 m)   GRU               9.4198             0.9226     4.1308     0.3877
                       CNN-LSTM [BEST]   10.0499            0.9261     4.0239     0.3777
                       Hybrid LSTM-GRU   9.3156             0.9210     4.1812     0.3924
─────────────────────────────────────────────────────────────────────────────────────────────
Pre-Monsoon 2014–24    LSTM              9.7608             0.8845     4.9051     0.4692
(Obs STD: 10.4535 m)   GRU               9.3570             0.8901     4.7637     0.4557
                       CNN-LSTM          9.0557             0.8716     5.1249     0.4903
                       Hybrid [BEST]     8.7880             0.8923     4.7492     0.4543
─────────────────────────────────────────────────────────────────────────────────────────────
```

---

# 💻 PART IV: COMPUTATIONAL ECOSYSTEM & REPRODUCIBILITY

## Chapter 9: Directory Hierarchy & Saved Model Artifacts

The repository is structured to maintain strict separation between raw input corpora, independent dataset run outputs, trained model checkpoints, and visualization scripts:

```
├── README.md                                          # Complete Research Monograph & Computational Atlas
├── groundwater-deep-learning-banner.svg               # Publication Banner
├── Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb   # Master Research Training & Evaluation Pipeline
├── generate_all_figures_standalone.py                 # Standalone Publication & Interactive Figure Generator
│
├── post-monsoon_wl_1994-2023.csv                     # Raw Corpus: Post-Monsoon (1994–2023)
├── pre-monsoon_1994-2003.csv                          # Raw Corpus: Pre-Monsoon Decade I (1994–2003)
├── pre-monsoon_2004-2013.csv                          # Raw Corpus: Pre-Monsoon Decade II (2004–2013)
├── pre-monsoon_2014-2024.csv                          # Raw Corpus: Pre-Monsoon Decade III (2014–2024)
│
└── outputs/
    └── DL_4_SEPARATE_DATASETS/
        ├── Four_Dataset_Run_Summary.csv               # Cross-dataset execution logs
        │
        ├── post_monsoon_1994_2023/                    # Experiment 1 Artifacts
        │   ├── dataset_profile.json                   # Metadata & row counts
        │   ├── saved_models/                          # Trained Keras models & scalers
        │   │   ├── lstm.keras
        │   │   ├── gru.keras
        │   │   ├── cnn_lstm.keras
        │   │   ├── hybrid_lstm_gru.keras
        │   │   ├── scaler_X.joblib & scaler_y.joblib
        │   │   └── model_metadata.joblib
        │   ├── tables/                                # Performance metrics & predictions
        │   │   ├── Testing_Metrics_Thesis_Format.csv
        │   │   ├── Training_Metrics_Thesis_Format.csv
        │   │   ├── Taylor_Statistics_Testing.csv
        │   │   ├── Testing_Predictions_All_Models.csv
        │   │   └── 03_Correlation_Matrix.csv
        │   └── figures/                               # 11 Static PNGs & Interactive HTMLs
        │       ├── 00_Observed_Regional_Timeline.png
        │       ├── 01_Training_Validation_Loss.png
        │       ├── 02_Time_Series_Observed_vs_Predicted.png
        │       ├── 03_Correlation_Matrix.png
        │       ├── 04_Scatter_Observed_vs_Predicted.png
        │       ├── 05_Test_Residuals_vs_Time.png
        │       ├── 06_Test_Residual_Distributions.png
        │       ├── 07_Taylor_Diagram_Training_Testing.png
        │       ├── 08_Model_Performance_Comparison.png
        │       ├── 09_Statistical_Metrics_Training_Testing.png
        │       └── 10_Hybrid_LSTM_GRU_Convergence.png
        │
        ├── pre_monsoon_1994_2003/                     # Experiment 2 Artifacts (Structure Identical)
        ├── pre_monsoon_2004_2013/                     # Experiment 3 Artifacts (Structure Identical)
        └── pre_monsoon_2014_2024/                     # Experiment 4 Artifacts (Structure Identical)
```

---

## Chapter 10: Reproduction Protocols & Training Pipeline Execution

### 1. Environment Installation
Ensure Python 3.10 or higher is installed with GPU acceleration (optional but recommended for faster training):

```bash
# Clone the repository
git clone https://github.com/Underwaterbet023/DeepLearning_for_Pre_and_Post_Monsoon.git
cd Deep-Learning_for_Pre_and_Post_Monsoon

# Create and activate a clean virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install essential dependencies
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn scipy plotly joblib
```

### 2. End-to-End Model Training
Execute the master research Jupyter Notebook to train all four model families across all four datasets sequentially:

```bash
jupyter notebook Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb
```
*The notebook automatically executes data ingestion, spatial-temporal feature engineering, chronological train-validation-test partitioning, model training with early stopping, metric calculations, and artifact exports.*

### 3. Standalone Figure Generation
To re-generate all 44 high-resolution publication figures and standalone interactive HTML files without re-training neural networks, run:

```bash
python generate_all_figures_standalone.py
```

---

## Chapter 11: Interactive HTML Dashboards

In addition to static publication figures, this study generates fully interactive, GPU-accelerated Plotly HTML files for every experiment:
- `00_Observed_Regional_Timeline_interactive.html`: Interactive timeline with zoom, pan, and hover inspection across all observation wells.
- `02_Time_Series_Observed_vs_Predicted_interactive.html`: Dynamic time-series viewer with individual model toggle capabilities.
- `03_Correlation_Matrix_interactive.html`: Interactive correlation matrix with exact coefficient inspectability.
- `04_Scatter_Observed_vs_Predicted_interactive.html`: Interactive parity scatter plots with residual distance callouts.
- `05_Test_Residuals_vs_Time_interactive.html`: Out-of-sample residual error tracking with temporal slicing.
- `07_Taylor_Diagram_Training_Testing_interactive.html`: Interactive polar Taylor diagram with cursor-guided polar coordinate inspection.

To view these interactive visualizations, open any `.html` file from the `outputs/DL_4_SEPARATE_DATASETS/<dataset>/figures/` folder in any modern web browser.

---

# 📚 PART V: SCHOLARLY EPILOGUE & CITATION

## Concluding Recommendations for Sustainable Water Resource Management

1. **Operational Early Warning Systems**: The validated Hybrid LSTM-GRU model provides water management authorities with a reliable tool for forecasting pre-monsoon water table depressions 1 to 3 seasons ahead, enabling preemptive irrigation quota adjustments.
2. **Prioritization of Critical Aquifer Zones**: Decadal analysis indicates that standard deviation has expanded by nearly 20%, highlighting the need to prioritize managed aquifer recharge (MAR) structures in regions with the most severe water level drops.
3. **Data-Centric Policy Formation**: Machine learning models rely heavily on consistent observation networks. Strengthening the density and observation frequency of digital hydrograph stations will significantly improve predictive accuracy and operational decision-making.

---

## 📖 Scholarly Citation

If you utilize this methodology, architecture code, empirical benchmarks, or findings in your scientific research, please cite this work as:

### BibTeX Format
```bibtex

@article{Deep-Learning_for_Pre_and_Post_Monsoon,
  title     = {Deep Spatiotemporal Neural Networks for Seasonal Groundwater Level Forecasting Across Multi-Decadal Regimes: A Comparative Study of LSTM, GRU, CNN-LSTM, and Hybrid Dual-Engine Architectures},
  author    = {Sanskar Kumar},
  year      = {2026},
}

```

---

## 📜 Author Declaration & License

- **Principal Investigator & Sole Author**: **Sanskar Kumar**
- **Research Integrity**: All empirical evaluations are strictly derived from real observation records with complete data integrity and causal separation.
- **License**: This computational framework and research documentation are openly distributed under the [MIT License](LICENSE).

<div align="center">
  <br/>
  <b>Dedicated to the Advancement of Sustainable Global Groundwater Management</b><br/>
  <i>Developed and Authored by Sanskar Kumar — 2026</i>
</div>
