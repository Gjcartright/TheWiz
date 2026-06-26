# QUANTIZED DYDX STATISTICAL ARBITRAGE AGENT MEMORY

## PROJECT OBJECTIVE

Build a production-grade statistical arbitrage trading system for dYdX that identifies and trades mean-reverting crypto pairs using:

- Cointegration
- Z-Score
- Hedge Ratio
- Half-Life
- ECM (Error Correction Model)
- Copula Dislocation Analysis
- Machine Learning Trade Scoring
- Automated Risk Management

The system must eventually operate autonomously while remaining fully observable and controllable.

---

# SUCCESS CRITERIA

Primary Success Metric:

Profit Factor >= 1.80

Secondary Metrics:

Win Rate >= 55%

Maximum Drawdown <= 20%

Sharpe Ratio > 1.5

No single pair contributes >10% of total drawdown

All executions logged

No orphaned positions

No unhedged exposure

---

# CORE TRADING THESIS

Markets contain temporary pricing inefficiencies between related assets.

When statistically related assets diverge beyond normal behavior:

1. Open long position on undervalued asset
2. Open short position on overvalued asset
3. Wait for spread reversion
4. Close positions
5. Capture convergence profit

The objective is NOT directional prediction.

The objective IS spread mean reversion.

---

# CURRENT STRATEGY STACK

## Layer 1

Cointegration

Purpose:

Identify pairs whose spread demonstrates long-term equilibrium.

Metrics:

- Cointegration P Value
- Test Statistic
- Critical Value

Requirements:

P Value < 0.05

---

## Layer 2

Spread Calculation

Formula:

Spread = Asset1 - (HedgeRatio × Asset2)

Purpose:

Create tradable spread series.

---

## Layer 3

Hedge Ratio

Purpose:

Normalize pair sizing.

Methods:

- OLS Regression
- Dynamic Hedge Ratio (future)

Output:

Position sizing relationship between both assets.

---

## Layer 4

Half Life

Purpose:

Estimate expected mean reversion speed.

Preferred Range:

0 < HalfLife < 25

Avoid:

Negative Half Life

Very large Half Life

Trending spreads

---

## Layer 5

Z-Score

Purpose:

Measure spread deviation from mean.

Current Entry:

Long Spread:

Z <= -1.5

Short Spread:

Z >= +1.5

Current Exit:

Z returns toward mean

Future:

Dynamic thresholds by market regime.

---

# COPULA MODEL

Purpose:

Measure extreme dislocations.

Role:

Tail Risk Monitor

Additional Entry Confirmation

Features:

- Copula Probability
- Tail Dependence
- Dislocation Magnitude

Use Cases:

Identify statistical arbitrage opportunities missed by Z-score alone.

---

# ECM MODEL

Error Correction Model

Purpose:

Measure strength of mean reversion.

Metrics:

- ECM(X)
- ECM(Y)
- ECM Strength

Interpretation:

Higher ECM Strength:

Faster expected reversion

Lower ECM Strength:

Avoid trade

---

# MACHINE LEARNING LAYER

Purpose:

Score trade quality.

Inputs:

- Z Score
- Half Life
- Hedge Ratio
- Cointegration P Value
- Spread Volatility
- ECM Strength
- Copula Probability
- Funding Rates
- Volume
- Liquidity
- Historical Pair Performance

Outputs:

Trade Score

0-100

Decision:

Trade / No Trade

Recommended Size

Confidence Score

---

# MODEL SELECTION

Current Preference:

LightGBM

Alternative:

XGBoost

Future:

Quantized ONNX Model

Reason:

Fast

Explainable

Low Resource Usage

Excellent Tabular Performance

---

# RISK MANAGEMENT

Maximum Risk Per Trade:

1%

Maximum Open Exposure:

10%

Maximum Pair Exposure:

5%

Daily Loss Limit:

3%

Weekly Loss Limit:

10%

Emergency Shutdown:

Triggered if:

- Exchange API Failure
- Execution Failure
- Missing Hedge Leg
- Database Failure
- Unexpected Drawdown

---

# EXECUTION LOGIC

Entry Conditions:

Cointegration Pass

Half Life Pass

Z Score Triggered

ECM Pass

Copula Pass

ML Score Pass

Risk Engine Pass

Execute Trade

---

# EXECUTION FAILSAFE

If Order 1 Fills

AND

Order 2 Fails

THEN

Immediately Close Order 1

IF Close Fails

Abort Program

Send Alert

Require Human Intervention

---

# DATABASE REQUIREMENTS

Store:

Trade History

Signals

Model Predictions

Spread History

Cointegration Results

Copula Results

ECM Results

Account Metrics

System Events

---

# FUTURE ROADMAP

Phase 1

Rebuild dYdX Bot

Phase 2

Backtesting Engine

Phase 3

Cointegration Scanner

Phase 4

ECM Integration

Phase 5

Copula Integration

Phase 6

Machine Learning Scoring

Phase 7

Walk Forward Validation

Phase 8

Portfolio Optimization

Phase 9

Production Deployment

Phase 10

Multi-Exchange Arbitrage

---

# IMPORTANT PRINCIPLES

Never trade because of opinion.

Never trade because of news.

Never trade because of emotion.

Trade only when:

Statistics agree.

Risk agrees.

Model agrees.

Execution agrees.

Protect capital first.

Profit second.

Survive long enough to compound.

---

# PROJECT OWNER PREFERENCES

Exchange:

dYdX

Strategy Type:

Statistical Arbitrage

Primary Filters:

Cointegration
Half Life
Z Score
ECM
Copula

Desired System:

Quantized AI Trading Agent

Target:

Institutional Grade Architecture

Target Profit Factor:

1.8+

Maximum Drawdown:

20% or less

Primary Goal:

Fully automated statistical arbitrage platform with AI-assisted trade selection and risk management.