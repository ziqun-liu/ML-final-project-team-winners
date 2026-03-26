# 6140 Project — Giudici et al. Research Notes

---

## 1. The U-Shape Curve Phenomenon

The U-shaped curve describes the **intraday distribution of trading volume over time**.

Trading volume is extremely high at market open (9:30 AM), steadily declines through midday (reaching its lowest point around 1:00 PM), then gradually recovers and peaks again at market close (4:00 PM). The overall shape resembles the letter "U".

**Why does the U-shape exist?**
- Overnight information (macro data releases, corporate announcements) accumulates before the open and gets digested all at once, triggering heavy trading activity.
- Midday sees reduced information flow and institutional inactivity, resulting in low volume.
- Before close, institutional investors perform portfolio rebalancing, causing volume to surge again.

**The core contribution of Giudici et al.** is discovering a puzzling pattern *beyond* the U-shape: systematic trading volume spikes occurring at every 5-minute interval throughout the day, even after controlling for open/close effects.

---

## 2. Moving Beyond Linear Models

### Proposed Model Progression: Logistic → Random Forest → Gradient Boosting → LSTM

This progression is **logically viable but requires a clearly defined research objective**, because these model types answer fundamentally different questions.

- **Logistic Regression**: Requires a binary dependent variable (e.g., "Is this a 5-minute node: yes/no?"). Giudici's original dependent variable is continuous (VOL_t as a percentage), so the problem must be redefined before logistic regression applies.
- **Random Forest / Gradient Boosting**: Support both regression and classification. Can automatically capture nonlinear relationships and feature interactions (e.g., the interaction between 5-minute dummies and Wednesdays) — a key advantage over OLS.
- **LSTM**: Designed for sequential data, capable of capturing long-term dependencies. Suitable for modeling the serial structure of trading volume, though overfitting risk is significant with noisy high-frequency financial data.

**The critical issue**: Giudici's OLS regression has clear economic interpretability — each β coefficient directly corresponds to an economic hypothesis. Switching to black-box models requires a tool to restore interpretability: **SHAP values**.

---

## 3. What Are SHAP Values?

SHAP (SHapley Additive exPlanations) is a tool for **explaining black-box model predictions**. It answers: for a specific prediction, how much did each feature contribute?

Its theoretical foundation comes from **Shapley values in cooperative game theory**. Intuitively, if the model is a "game" and each feature is a "player," SHAP computes each player's marginal contribution to the final prediction, while guaranteeing that all feature contributions sum to the difference between the prediction and the baseline (efficiency axiom).

**In the context of this research**, SHAP values can be applied as follows: after training a Random Forest to predict VOL_t, use SHAP to analyze how much the MIN_01 dummy variable boosted the prediction, and whether its contribution is larger in the options market than in the spot market.

**Three common SHAP visualizations**:
- **Summary Plot**: Global feature importance across all observations
- **Dependence Plot**: Relationship between a specific feature value and its SHAP contribution
- **Force Plot**: Decomposition of a single observation's prediction

---

## 4. Allocation, Price/Volume Divergence, and Volatility Forecasting

### 4.1 How Much to Allocate During Trading

**Kelly Criterion** (classical solution):

$$f^* = \frac{bp - q}{b}$$

where $f^*$ is the fraction of capital to allocate, $b$ is the expected return per unit risk, $p$ is the probability of winning, and $q = 1 - p$. Full Kelly is theoretically optimal for long-run growth but extremely aggressive in practice. Most practitioners use **Half-Kelly or Quarter-Kelly**.

**Risk-based allocation**:

$$\text{Position Size} = \frac{\text{Risk Budget per Trade}}{\text{Expected Volatility of Asset}}$$

**Relevance to the 5-minute effect**: During volume spikes at 5-minute intervals, liquidity is higher and market impact cost is lower. A rational allocation strategy would therefore **increase position size at these intervals** because the same order moves the price less.

---

### 4.2 Creating Returns from Price/Volume Divergence

Price/volume divergence occurs when price and volume move in opposite directions from what theory predicts (Karpoff 1987).

| Price | Volume | Interpretation | Signal |
|-------|--------|----------------|--------|
| Rising | Falling | Weak rally, lacks conviction | Short |
| Falling | Falling | Weak selloff, likely stabilization | Long |
| Rising | Rising | Confirmed trend | Follow trend |
| Falling | Rising | Panic selling / capitulation | Contrarian long |

**Implementation using Giudici's framework**: Compute the residual from Model 2 ($\hat{\varepsilon}_t$ = actual volume minus predicted volume). A large positive residual with no price move suggests informationally interesting activity. A significant price move with a small or negative residual suggests an algorithmic artifact rather than genuine information.

---

### 4.3 Volatility Forecasting

**GARCH(1,1)** — classical baseline:

$$\sigma^2_t = \omega + \alpha \cdot \varepsilon^2_{t-1} + \beta \cdot \sigma^2_{t-1}$$

where $\varepsilon^2_{t-1}$ is the ARCH term (past shock) and $\sigma^2_{t-1}$ is the GARCH term (volatility persistence). GJR-GARCH and EGARCH extend this to capture the asymmetric leverage effect.

**HAR-RV** — particularly relevant for intraday data:

$$RV_{t+1} = \beta_0 + \beta_d \cdot RV_t^{(d)} + \beta_w \cdot RV_t^{(w)} + \beta_m \cdot RV_t^{(m)} + \varepsilon_{t+1}$$

**Connection to the 5-minute effect**: Volume spikes at 5-minute intervals likely inflate RV estimates computed from raw minute data. A cleaner approach: subsample at non-spike minutes or explicitly control for the 5-minute dummy in RV computation.

---

## 5. All Models Used in Giudici et al.

### Model 1 — Baseline: Minute-by-Minute Volume Coefficients

$$VOL_t = \alpha \cdot VOL_{t-1} + \sum_{min=0}^{59} \beta_{min} M_{min} + \varepsilon_t$$

**Variable definitions**:
- $VOL_t$: trading volume at minute $t$ as a percentage of total daily volume
- $VOL_{t-1}$: first-order autoregressive term, controls serial correlation
- $M_{min}$: dummy variable equal to 1 when the time is minute $min$ of each hour (e.g., $M_1 = 1$ at 10:01, 11:01, 12:01, etc.)
- 60 dummy variables total; results visualized as 95% confidence interval plots (Figures 2A, 2B)

---

### Model 2 — Simplified Model: Group Test for 5-Minute Effects (Core Model)

$$VOL_t = \alpha \cdot VOL_{t-1} + \beta_1 + \beta_2 MIN\_01_t + \beta_3 MIN\_31_t + \beta_4 MIN\_16\_46_t + \beta_5 MIN\_k5_t + \varepsilon_t$$

| Variable | Takes Value of 1 At | Economic Interpretation |
|----------|-------------------|------------------------|
| $MIN\_01_t$ | hh:01 | One minute past the hour (near market close 4:00 PM) |
| $MIN\_31_t$ | hh:31 | One minute past the half hour (near market open 9:30 AM) |
| $MIN\_16\_46_t$ | hh:16, hh:46 | Quarter-hour marks |
| $MIN\_k5_t$ | hh:06, hh:11, hh:21, hh:26, hh:36, hh:41, hh:51, hh:56 | Remaining 5-minute interval marks |

**Benchmark group**: All other minutes, represented by the constant $\beta_1$. Each $\beta$ coefficient measures excess trading volume percentage at that group of minutes **relative to ordinary minutes**.

---

### Model 3a — Open/Close Interaction Model

$$\begin{aligned} VOL_t = \; &\alpha \cdot VOL_{t-1} + \beta_1 + \beta_2 MIN\_01_t + \beta_3 MIN\_31_t + \beta_4 MIN\_16\_46_t + \beta_5 MIN\_k5_t \\ &+ O\_C_t \left(\beta_5 MIN\_01_t + \beta_6 MIN\_31_t + \beta_7 MIN\_16\_46_t + \beta_8 MIN\_k5_t\right) + \varepsilon_t \end{aligned}$$

- $O\_C_t = 1$ during the first hour (9:30–10:30) and last hour (15:00–16:00) of trading, 0 otherwise
- Interaction term coefficients capture the **additional** 5-minute spike during open/close periods

---

### Model 3b — Wednesday Interaction Model

$$\begin{aligned} VOL_t = \; &\alpha \cdot VOL_{t-1} + \beta_1 + \beta_2 MIN\_01_t + \beta_3 MIN\_31_t + \beta_4 MIN\_16\_46_t + \beta_5 MIN\_k5_t \\ &+ WED_t \left(\beta_5 MIN\_01_t + \beta_6 MIN\_31_t + \beta_7 MIN\_16\_46_t + \beta_8 MIN\_k5_t\right) + \varepsilon_t \end{aligned}$$

- $WED_t = 1$ on Wednesdays, 0 otherwise
- Economic background: FOMC typically releases interest rate decisions on Wednesdays at 2:00 PM ET (i.e., hh:01)

---

### Model 3c — Negative Return Interaction Model

$$\begin{aligned} VOL_t = \; &\alpha \cdot VOL_{t-1} + \beta_1 + \beta_2 MIN\_01_t + \beta_3 MIN\_31_t + \beta_4 MIN\_16\_46_t + \beta_5 MIN\_k5_t \\ &+ NEG\_RET_t \left(\beta_5 MIN\_01_t + \beta_6 MIN\_31_t + \beta_7 MIN\_16\_46_t + \beta_8 MIN\_k5_t\right) + \varepsilon_t \end{aligned}$$

- $NEG\_RET_t = 1$ when the ETF return at minute $t$ is negative, 0 otherwise
- Key finding: For leveraged ETFs (DDM, DOG), interaction term coefficients are all significantly positive, meaning the 5-minute effect is **almost entirely driven by negative returns**

---

### Model Progression Summary

```
Model 1  →  Full description (60 dummy variables)
    ↓
Model 2  →  Compressed into 4 groups of 5-minute nodes (core test)
    ↓
Model 3a →  Controls for open/close effect (spurious correlation?)
Model 3b →  Controls for Wednesday effect (FOMC-driven?)
Model 3c →  Controls for negative return effect (sentiment-driven?)
```

Each step **progressively eliminates alternative explanations**. The final conclusion: even after controlling for all these factors, the 5-minute effect persists, supporting the algorithmic trading bias hypothesis.

---

## 6. Data Construction: VOL_t and the AR Term

### VOL_t — Two-Step Construction

**Step 1**: Obtain raw minute-level trading volume (source: TickData)

| Timestamp | Raw Volume (shares) |
|-----------|-------------------|
| 2020-01-02 09:31 | 8,432,100 |
| 2020-01-02 09:32 | 3,201,400 |
| ... | ... |

**Step 2**: Divide by total daily volume

$$VOL_t = \frac{\text{Volume at minute } t}{\sum_{i=09:31}^{16:00} \text{Volume}_i}$$

$$VOL_{09:31} = \frac{8,432,100}{4,500,000,000} = 0.001874 = 0.187\%$$

**Why normalize?**
1. Eliminates cross-ETF size differences (SPY volume is ~100× DDM)
2. Removes day-of-week effects (Wednesday volume is systematically higher)

---

### VOL_{t-1} — The AR Term

The AR term requires no additional data collection — it is simply the same column shifted by one row:

```python
df['VOL_lag1'] = df['VOL'].shift(1)
```

**Why include the AR term?** Trading volume exhibits positive autocorrelation — it declines from open to midday and rises from midday to close. Without the AR term, residuals would be severely autocorrelated, making OLS standard errors unreliable (understated), leading to inflated t-statistics.

**Important**: The first observation of each trading day (09:31) must have $VOL_{t-1}$ set to NaN to avoid cross-day contamination.

---

## 7. Example Observation (Single Row) for Model 2

### 10:31 AM on a Tuesday, SPY

| Field | Value | Explanation |
|-------|-------|-------------|
| $t$ | Tuesday 10:31 AM | Timestamp |
| $VOL_t$ | 0.00187 | 0.187% of that day's total SPY volume |
| $VOL_{t-1}$ | 0.00210 | 0.210% traded at 10:30 AM |
| $MIN\_01_t$ | 0 | Not minute :01 |
| $MIN\_31_t$ | **1** | This is minute :31 → dummy fires |
| $MIN\_16\_46_t$ | 0 | Not minute :16 or :46 |
| $MIN\_k5_t$ | 0 | Not a remaining 5-minute mark |

**Model prediction**:

$$\hat{VOL}_t = \alpha \cdot (0.00210) + \beta_1 + \beta_3 \cdot (1)$$

From Table 3 Panel D of Giudici, $\beta_3 \approx 0.00047$ for SPY — meaning :31 minutes carry roughly **0.047 percentage points of extra daily volume share** compared to ordinary minutes.

### Contrasting Row: 10:33 AM, Same Day

All dummies = 0. This is a benchmark minute:

$$\hat{VOL}_t = \alpha \cdot (0.00187) + \beta_1$$

The difference between 10:31 and 10:33 predictions (holding $VOL_{t-1}$ constant) is exactly $\beta_3$ — this is how OLS isolates the 5-minute effect.

---

## 8. Realized Volatility (RV) — Definition and Theory

### Intuition

Traditional volatility uses daily returns (one data point per day). With minute-level data, we have 390 observations per day — using all of them produces a far more accurate volatility estimate. This is the motivation for Realized Volatility.

### Formal Definition

**Step 1**: Compute per-minute log returns

$$r_{t,i} = \ln(P_{t,i}) - \ln(P_{t,i-1})$$

**Step 2**: Sum squared returns across the day

$$RV_t = \sum_{i=1}^{M} r_{t,i}^2$$

where $M = 390$ for standard US equity markets (9:30–16:00).

### Theoretical Foundation

In continuous-time finance, price follows a diffusion process:

$$dP_t = \mu_t \, dt + \sigma_t \, dW_t$$

The true (unobservable) **Integrated Variance** is:

$$IV = \int_0^1 \sigma_t^2 \, dt$$

Andersen & Bollerslev (1998) proved that as sampling frequency increases:

$$RV_t \xrightarrow{p} IV$$

RV is a **consistent estimator** of true integrated variance.

### Numerical Example

| Minute | Price | Log Return $r$ | $r^2$ |
|--------|-------|---------------|-------|
| 9:31 | 400.00 | — | — |
| 9:32 | 400.80 | +0.0020 | 0.00000400 |
| 9:33 | 400.40 | −0.0010 | 0.00000100 |
| 9:34 | 401.20 | +0.0020 | 0.00000400 |
| 9:35 | 400.80 | −0.0010 | 0.00000100 |

$$RV = 0.000004 + 0.000001 + 0.000004 + 0.000001 = 0.000010$$

$$\text{Annualized Volatility} = \sqrt{RV \times 252} \approx 5.02\%$$

### Key Properties

- **Direction-neutral**: Both positive and negative returns contribute positively (squared)
- **Additivity**: Weekly RV = sum of 5 daily RVs (foundation of HAR-RV's weekly term)
- **Sampling frequency tradeoff**: Higher frequency is theoretically better, but **microstructure noise** contaminates ultra-high-frequency data. 5-minute sampling is the most common choice in empirical literature — which intersects directly with Giudici's 5-minute effect

---

## 9. HAR-RV Model — Full Explanation

### Why Three Time Scales?

**GARCH's limitation**: GARCH(1,1) has only one memory dimension — yesterday's volatility. It implicitly assumes all market participants operate on the same time horizon.

**HAR-RV's core assumption**: Markets contain three types of participants with fundamentally different time horizons, each contributing a distinct component to volatility formation.

### Three Time Scales

**Daily** $RV_t^{(d)}$:

$$RV_t^{(d)} = RV_t$$

Represents **intraday traders and high-frequency algorithms**. Most sensitive to yesterday's events. In your research context, the 5-minute spike is primarily a daily-scale phenomenon.

**Weekly** $RV_t^{(w)}$:

$$RV_t^{(w)} = \frac{1}{5} \sum_{i=1}^{5} RV_{t-i}$$

Represents **short-term institutional investors and trend followers**. Evaluate market conditions on a weekly basis.

**Monthly** $RV_t^{(m)}$:

$$RV_t^{(m)} = \frac{1}{22} \sum_{i=1}^{22} RV_{t-i}$$

Represents **pension funds and long-horizon institutions**. Insensitive to short-term fluctuations but responsive to volatility regime shifts.

### Full Model

$$RV_{t+1} = \beta_0 + \beta_d \cdot RV_t^{(d)} + \beta_w \cdot RV_t^{(w)} + \beta_m \cdot RV_t^{(m)} + \varepsilon_{t+1}$$

### Empirical Pattern of Coefficients

$$\beta_d > \beta_w > \beta_m > 0$$

All three coefficients are significantly positive but decreasing in magnitude. Recent information has more predictive power, but distant information still makes an **independent, irreplaceable contribution** — this is the manifestation of **long memory in volatility**, which GARCH(1,1) cannot capture.

### Connection to the 5-Minute Effect

The algorithmic spike's contamination of RV is **asymmetric across time scales**:
- At the **daily scale**: spike impact is direct and largest
- At the **weekly scale**: diluted by 5-day averaging, but systematic bias persists if spikes occur every day (Giudici proves they do)
- At the **monthly scale**: further diluted but not eliminated

This means the difference between $RV^{clean}$ and $RV^{raw}$ is largest at the daily scale and smallest at the monthly scale — a testable hypothesis and a concrete contribution to the HAR-RV literature.

---

## 10. Research Extension: What Can Be Built on Giudici?

### What Giudici Already Solved

The paper proves that ETF trading volume exhibits systematic spikes at every 5-minute node, persisting after controlling for open/close effects, Wednesday effects, and negative returns. This supports the algorithmic trading bias hypothesis.

**What the paper does NOT answer**: Does the 5-minute effect influence price and volatility? Can it be predicted and exploited?

---

### Extension Direction 1: Contamination of Volatility Measurement

**Problem**: The standard RV computation is:

$$RV_t^{raw} = \sum_{i=1}^{390} r_{t,i}^2$$

Since Giudici proves algorithmic volume spikes at hh:01, hh:06, hh:11... inflate trading activity at these minutes, price may also be temporarily distorted, causing $r_{t,i}^2$ to be overstated at spike minutes. Raw RV **systematically overestimates true information-driven volatility**.

**Contribution**: Construct and compare two RV versions:

$$RV_t^{raw} = \sum_{i=1}^{390} r_{t,i}^2 \quad \text{(all minutes)}$$

$$RV_t^{clean} = \sum_{i \notin \{5\text{-min nodes}\}} r_{t,i}^2 \quad \text{(spike minutes excluded)}$$

If $RV^{clean}$ produces better HAR-RV forecast performance than $RV^{raw}$, this proves that the 5-minute algorithmic effect contaminates traditional volatility measurement.

---

### Extension Direction 2: HAR-RV with Algorithmic Intensity

**Problem**: Giudici studied volume only, not volatility. But the two are closely related.

Aggregate Model 2 residuals into a daily indicator:

$$SPIKE_t = \frac{1}{N_{spike}} \sum_{i \in \text{5-min nodes}} \hat{\varepsilon}_{t,i}$$

**Extended HAR-RV**:

$$RV_{t+1} = \beta_0 + \beta_d \cdot RV_t^{(d)} + \beta_w \cdot RV_t^{(w)} + \beta_m \cdot RV_t^{(m)} + \beta_{spike} \cdot SPIKE_t + \varepsilon_{t+1}$$

**Research question**: Does algorithmic trading intensity (measured by 5-minute spike magnitude) predict next-day volatility? If $\beta_{spike}$ is significant, algorithmic behavior influences not just volume distribution but overall market risk levels.

---

### Extension Direction 3: ML + SHAP Decomposition

Use LSTM or Gradient Boosting to predict $RV_{t+1}$ with features:

$$\mathbf{X}_t = \left[RV_t^{(d)},\ RV_t^{(w)},\ RV_t^{(m)},\ SPIKE_t,\ NEG\_RET_t,\ \hat{\varepsilon}_t^{(k5)} \right]$$

Apply **SHAP values** to decompose each prediction:

$$\hat{RV}_{t+1} = \phi_0 + \underbrace{\phi_{RV^{(d)}} + \phi_{RV^{(w)}} + \phi_{RV^{(m)}}}_{\text{information-driven component}} + \underbrace{\phi_{SPIKE} + \phi_{NEG\_RET}}_{\text{algorithmic behavior component}}$$

This quantifies: **what fraction of next-day volatility is explained by algorithmic artifacts versus genuine information?**

---

### Logical Flow of Extensions

```
Giudici: 5-minute spike exists (volume level)
        ↓
Direction 1: Spike contaminates RV measurement (measurement level)
        ↓
Direction 2: Cleaned RV + spike intensity predicts volatility (forecasting level)
        ↓
Direction 3: ML + SHAP decomposes algorithmic vs. information components (explanation level)
```

From **phenomenon discovery → measurement correction → predictive application → mechanism explanation**.

---

## 11. Mission & Vision

### Mission

**Quantify how systematic biases from algorithmic trading distort the measurement and prediction of market volatility.**

Specifically: by identifying and stripping the algorithmic-driven component (the 5-minute spike) from ETF intraday trading volume, construct a cleaner volatility measure, and test whether algorithmic trading intensity has predictive power over market risk levels.

### Vision

**The current market paradox**: Modern financial risk management relies heavily on volatility forecasting, but volatility measurement itself has been contaminated by algorithmic trading. The tools we use to "measure risk" are mixed with algorithmic noise unrelated to genuine information. This is like measuring distance with a systematically worn ruler — every measurement is biased, and you don't even know by how much.

> Build an analytical framework capable of distinguishing **algorithm-noise-driven volatility** from **information-driven true volatility**, providing investors, risk managers, and regulators with more accurate market risk signals.

### One-Sentence Version

> Our research aims to answer: **In algorithm-dominated modern markets, how much of the volatility we measure is a genuine risk signal, and how much is merely the echo of algorithms?**

---

## 12. Full Research Pipeline

### Step 1: OLS — Replicate Giudici Model 2

$$VOL_t = \alpha \cdot VOL_{t-1} + \beta_1 + \beta_2 MIN\_01_t + \beta_3 MIN\_31_t + \beta_4 MIN\_16\_46_t + \beta_5 MIN\_k5_t + \varepsilon_t$$

**Output**: Regression coefficient table confirming 5-minute effect significance
**Byproduct**: Residuals $\hat{\varepsilon}_t$ — the algorithmic noise indicator

---

### Step 2: OLS — Extended HAR-RV

$$RV_{t+1} = \beta_0 + \beta_d \cdot RV_t^{(d)} + \beta_w \cdot RV_t^{(w)} + \beta_m \cdot RV_t^{(m)} + \beta_{spike} \cdot SPIKE_t + \varepsilon_{t+1}$$

**Output**: Coefficient table; key question is whether $\beta_{spike}$ is statistically significant
**Decision**: If significant → proceed to Step 3

---

### Step 3A: GBM / LSTM — Predict $RV_{t+1}$

**Input features**:

$$\mathbf{X}_t = \left[RV_t^{(d)},\ RV_t^{(w)},\ RV_t^{(m)},\ SPIKE_t,\ NEG\_RET_t \right]$$

**Output**: Prediction accuracy comparison + SHAP decomposition plots

| Model | $R^2$ | MSE | Notes |
|-------|-------|-----|-------|
| HAR-RV (OLS) | baseline | baseline | Linear benchmark |
| HAR-RV + GBM | higher? | lower? | Captures nonlinearity |
| HAR-RV + LSTM | higher? | lower? | Captures long memory |

---

### Model Comparison Summary

| | OLS | GBM | LSTM |
|--|-----|-----|------|
| Output | $\beta$ coefficients + t-stats | Accuracy metrics + SHAP | Accuracy metrics + SHAP |
| Interpretability | Highest (direct) | Medium (requires SHAP) | Low (requires SHAP) |
| Captures nonlinearity | No | Yes | Yes |
| Captures temporal dependency | Limited (AR term only) | No | Yes |
| Academic role | Establish baseline, test significance | Improve forecast accuracy | Capture long-memory structure |

**OLS establishes an interpretable baseline; GBM improves predictive accuracy and reveals nonlinearity via SHAP; LSTM captures the long-memory structure of volatility.**

---

## 13. Research Conclusions (if Step 3A is Selected)

### Finding 1: The 5-Minute Effect Predicts Next-Day Volatility (from Step 2)

Algorithmic trading behavior leaves a measurable footprint at the volume level, and this footprint has statistically significant predictive power over next-day market volatility.

> Algorithmic trading's systematic bias leaves measurable traces in trading volume patterns; these traces have statistically significant predictive power over next-day market realized volatility.

---

### Finding 2: The Relationship is Nonlinear (from Step 3A accuracy comparison)

If GBM/LSTM achieves significantly higher $R^2$ than OLS-HAR-RV, the relationship between $SPIKE_t$ and $RV_{t+1}$ is **not simply linear** — OLS underestimates the predictive value of algorithmic activity.

| Market Condition | Effect of $SPIKE_t$ | Economic Interpretation |
|-----------------|-------------------|------------------------|
| Low-volatility environment | Small impact on $RV_{t+1}$ | Spikes are pure noise in calm markets |
| High-volatility environment | Large impact on $RV_{t+1}$ | Algorithmic activity amplifies risk in turbulent markets |
| Negative-return days | Stronger positive contribution | Consistent with Giudici's NEG_RET finding |

> Traditional linear models systematically underestimate the amplifying effect of algorithmic trading on risk during high-volatility environments.

---

### Finding 3: SHAP Quantifies Algorithmic vs. Information Components (from Step 3A SHAP)

$$\hat{RV}_{t+1} = \phi_0 + \underbrace{\phi_{RV^{(d)}} + \phi_{RV^{(w)}} + \phi_{RV^{(m)}}}_{\text{information-driven}} + \underbrace{\phi_{SPIKE} + \phi_{NEG\_RET}}_{\text{algorithmic behavior}}$$

Compute the percentage of the volatility forecast explained by algorithmic components across all trading days. For example:

> Over 2013–2021, the 5-minute algorithmic effect explains on average X% of SPY's next-day realized volatility forecast, rising to Y% during stress periods (e.g., March 2020).

---

### Contributions to Literature

**Extension of Giudici**: Giudici proved the existence and persistence of the 5-minute effect at the volume level. This research further demonstrates that the effect crosses into the volatility domain, with predictive power that is more pronounced in a nonlinear framework.

**Contribution to HAR-RV literature**: Traditional HAR-RV uses only three time scales of historical RV. This research shows that adding an algorithmic trading intensity indicator ($SPIKE_t$) significantly improves forecast accuracy — market microstructure information contains volatility-predictive content not yet captured by the HAR framework.

**Practical implication for risk management**: Algorithmic trading intensity is a **real-time observable leading indicator** — $SPIKE_t$ can be computed before market close and incorporated into next-day risk budgeting decisions, providing more timely information than historical RV alone.

---

### Honest Limitation

**Causality problem**: The fact that $SPIKE_t$ significantly predicts $RV_{t+1}$ may not mean algorithmic activity *causes* higher volatility. Both could be driven by a common factor (e.g., macroeconomic uncertainty simultaneously elevates algorithmic activity and next-day volatility). Neither OLS nor ML can resolve this causal identification problem — this must be explicitly acknowledged in the limitations section.

---

*End of Research Notes*
