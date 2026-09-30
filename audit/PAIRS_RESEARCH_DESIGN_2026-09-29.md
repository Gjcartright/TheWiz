# WIZ pairs research design review

**Date:** 2026-09-29
**Purpose:** Recommend a statistically defensible, computationally efficient research and acceptance process for the TheWiz crypto pairs project.
**Scope:** Design review only. No pair backtest, new collection, model training, or trading was performed.

## Recommendation

Use a **staged, portfolio-aware evidence gate**:

1. Keep the current acceptance-manifest requirements as a research-candidate gate. A pass means “eligible for further evaluation,” not “ready to trade.”
2. Treat the stricter `research.yaml` targets as a later promotion gate, evaluated on genuinely out-of-sample and after-cost results. Define explicitly whether trade counts and pair counts apply per pair or to the portfolio.
3. Add family-wide selection-bias controls and a locked prospective confirmation stage. Keep Testnet and live authority separate.

This gives each existing threshold set a defined job, avoids changing numeric thresholds before an owner decision, and prevents a weak exploratory pass from being treated as deployment approval. The mapping must be written into a versioned resolver before the run is accepted.

## What the research says

The results are mixed; they do not support importing a single “best” cointegration, distance, or copula recipe into this project.

| Evidence | Finding relevant to this project | Limit |
| --- | --- | --- |
| Gatev, Goetzmann, and Rouwenhorst, equity pairs study | The classic distance method found historical relative-value results using long equity histories and bootstrap checks. It is a useful baseline, not proof that a crypto-perpetual implementation works. [Paper](https://repec.som.yale.edu/icfpub/publications/2573.pdf) | Equities, long historical sample, different venue, costs, and market structure. |
| 2020 study of 26 Binance cryptocurrencies | Compared distance and cointegration at 5-minute, hourly, and daily frequencies. Reported sensitivity to parameters, transaction costs, and execution windows, with overall results varying against benchmarks. [IEEE paper](https://ieeexplore.ieee.org/document/9200323/) | One exchange, sample, and set of parameter choices; cannot determine TheWiz’s result. |
| 2024 study of Binance USDT-margined futures | Its conventional cointegration and copula comparisons lost net profitability under its modeled costs in several hourly/5-minute cases, while its proposed reference-asset method did better in that study. This is a strong warning to evaluate costs and turnover with each method, not a reason to adopt its method. [Paper](https://doi.org/10.1186/s40854-024-00702-7) | Specific 20-coin, 2021–2023 sample and study-specific execution assumptions. |
| Backtest selection research | Trying more pairs, signals, and parameter variants increases the chance of selecting an overfit winner. PBO/CSCV was proposed to assess that selection risk. [PBO paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) | A diagnostic on the tried strategy set; it does not replace chronological out-of-sample testing. |
| Deflated Sharpe Ratio | Adjusts Sharpe evidence for selection bias and non-normal returns. It is useful when many alternatives were tried, alongside the full trial log. [DSR paper](https://doi.org/10.2139/ssrn.2460551) | Does not fix look-ahead, bad costs, data errors, or a short sample. |
| White’s Reality Check | Tests whether the best model from a search has predictive superiority to a benchmark while accounting for data snooping. [Paper](https://doi.org/10.1111/1468-0262.00152) | Can be conservative and depends on a defensible benchmark and bootstrap setup. |

The practical conclusion is to use a simple baseline, strict chronological validation, realistic costs, and explicit multiple-testing accounting. More complex models only earn a place after that baseline survives.

## Quantopian repository review

The linked organization currently lists 97 repositories, most unrelated to this project. I reviewed the relevant public code references rather than treating the entire organization as pairs-strategy evidence.

| Repository/reference | Reusable lesson | Do not copy as a TheWiz strategy |
| --- | --- | --- |
| [`research_public` basic pairs template](https://github.com/quantopian/research_public/blob/master/template_algorithms/basic_pairs_trade_optimize_template.py) | Makes pair direction, entry/exit thresholds, target weights, rebalance timing, and gross leverage constraint visible in one example. That is a useful checklist for making an experiment specification concrete. | The file labels itself a learning example and warns against trading it as-is. It uses a hard-coded equity pair, a rolling difference of raw prices, a short/long moving-average z-score, and fixed equal-and-opposite weights. It does not establish cointegration, a beta hedge, crypto perpetual costs, robust execution, or out-of-sample profitability. Its thresholds and pair selection are not evidence for WIZ settings. |
| [`zipline`](https://github.com/quantopian/zipline) and [slippage models](https://github.com/quantopian/zipline/blob/master/zipline/finance/slippage.py) | Useful architecture precedent for a streaming event-driven simulator with explicit commission, slippage, order-volume limits, and fill behavior. The beginner tutorial describes costs and order delays as part of simulation. [Tutorial](https://github.com/quantopian/zipline/blob/master/docs/source/beginner-tutorial.rst) | Its native asset, calendar, volume, and commission assumptions are primarily equities/futures. The implementation cannot stand in for a crypto perpetual venue adapter, funding accrual, 24/7 bars, mark/index prices, liquidation, or two-leg partial-fill risk. |
| [`empyrical`](https://github.com/quantopian/empyrical) and [`pyfolio`](https://github.com/quantopian/pyfolio) | Useful references for organizing portfolio returns, drawdown, rolling metrics, and tear-sheet diagnostics. Empyrical exposes common metrics such as maximum drawdown and alpha/beta. | Recalculate and validate every convention for crypto: annualization at continuous 24/7 cadence, missing bars, funding-inclusive returns, leverage, and the portfolio equity denominator. Do not add either library solely because it is from Quantopian; use the project’s existing metric implementation unless a metric gap is confirmed. |

This code review reinforces the design recommendations but does not change the proposed strategy or approve any dependency. The strongest transferable idea is explicit event timing and execution-cost modeling; the example’s signal formula and thresholds should not be imported.

## Quantopian community material

I corrected the earlier incomplete search by using the user's authenticated Chrome session and the Quantopian site's own search interface. A search for **“pairs trading” returned 19 posts** (plus separate matches in comments, messages, lessons, events, and members). A search for **“crypto” returned 41 posts**. The site's search result counts are what the UI displayed during this review; they are not a guarantee that every post in every category, comment, or attachment was indexed.

Opened and read the following directly relevant posts:

| Post | Date | What the community post says | Audit treatment |
|---|---:|---|---|
| [Pairs Trading via Unsupervised Learning](https://community.quantopian.com/c/community-forums/pairs-trading-via-unsupervised-learning) | Nov 20, 2024 | Summarizes clustering methods (k-means, DBSCAN, agglomerative clustering), firm characteristics and historical returns, and a reported annualized Sharpe of 2.69. Links SSRN 3835692. | Candidate methods and a reported result only; independently reproduce paper protocol, data, costs, and selection-bias controls before considering it. |
| [Pairs Trading with Time-Series Deep Learning Models](https://community.quantopian.com/c/community-forums/pairs-trading-with-time-series-deep-learning-models) | Feb 17, 2026 | Describes LSTM/transformer and conventional ML models for residual forecasts on equities and crypto; links a 2025 *Journal of Finance and Data Science* paper. The publisher search result exposed an abstract reporting seven models, S&P 500 and cryptocurrency data over 2012–2020, and claimed higher returns/Sharpe, lower turnover, and lower drawdown versus a relative-value baseline. | High relevance, but the full publisher page triggered a human-verification challenge, so I could not inspect the paper's complete methods, costs, or PDF. Treat those results as abstract-level author claims pending full-paper review. |
| [Pairs Trading Using Clustering and Deep Reinforcement Learning](https://community.quantopian.com/c/community-forums/pairs-trading-using-clustering-and-deep-reinforcement-learning) | Dec 16, 2024 | Describes convolutional autoencoders for latent factors, clustering equity indices, and RL; reports up to 21.86% annualized returns for 2017–2022. Links SSRN 4504599. | Equity result, not crypto validation; inspect benchmarks, training/test separation, turnover, and financing costs. |
| [Mastering Pair Trading with Risk-Aware Recurrent Reinforcement Learning](https://community.quantopian.com/c/community-forums/mastering-pair-trading-with-risk-aware-recurrent-reinforcement-learning) | Aug 27, 2024 | Describes CREDIT (bidirectional GRU plus temporal attention and a risk-aware reward) and claims higher profits with less trading on U.S. stocks. Links arXiv 2304.00364. | Useful risk-aware-policy idea; an equities paper does not establish transfer to crypto or exchange execution. |
| [LLM powered Quant Research Workflows](https://community.quantopian.com/c/community-forums/llm-powered-quant-research-workflows) | Feb 16, 2024 | Author says a GPT-4 demo generated code to find potentially cointegrated Binance pairs; the thread has seven comments and a one-minute demo video. | Supports an LLM-assisted research workflow only. Generated strategy code and pair selection require deterministic review and independent validation; no trading-performance evidence in the post. |
| [Statistical Arbitrage within Crypto Markets using PCA](https://community.quantopian.com/c/community-forums/statistical-arbitrage-within-crypto-markets-using-pca) | Jun 28, 2025 | Applies PCA to crypto statistical arbitrage and OU residual modeling; the post itself flags few trades and weak mean reversion as limits to robustness. Links SSRN 5263475. | Most directly aligned crypto source found; the stated limitations argue for strict out-of-sample, stability, and trade-count gates. |
| [C++ Pairs Trading Examples with ChatGPT](https://community.quantopian.com/c/community-forums/c-pairs-trading-examples-with-chatgpt) | Apr 21, 2024 | C++/Kalman-filter example prompted by a community question. | Potential implementation reference; not evidence of a validated edge. |

The “pairs trading” results also surfaced contextual material on crypto market making, ETF spreads, a pair-finding community member, implied-volatility metrics, and research discussions that mention pairs. These are leads rather than direct evidence for TheWiz. The earlier crypto search also surfaced **Market Making in Crypto** (Hummingbot, SOL-USDT/DOGE-USDT/GALA-USDT, backtesting and claimed live trading), **In Crypto We Trend**, **Crypto Contagion**, **What Drives Crypto Asset Prices?**, and other crypto market-structure/risk posts. Their abstracts and summaries should not be treated as independently verified findings.

This corrects my earlier false negative: the authenticated site search did surface relevant crypto-pairs and cointegration posts. The review is still **targeted, not a complete capture of every post on the whole site**: the site's search returns many categories and 99+ community-forum items, and the current interface does not provide a reliable, complete export of all posts, comments, lessons, events, messages, and attachments. I manually reviewed the high-relevance results above; I have not exhaustively read all 19 “pairs trading” matches or all 41 “crypto” matches, nor every lesson/forum category. The [site Terms of Service](https://community.quantopian.com/terms) prohibit use of spiders, robots, crawlers, and data-mining tools, so this was performed through the site's own UI without bulk scraping. The visible sidebar includes Hummingbot market making, backtesting, statistics/econometrics, financial data, machine learning, and quantitative-finance research spaces; category presence does not validate a strategy. The [December 2023 Quantopian Lectures announcement](https://community.quantopian.com/c/announcements/quantopian-lectures-available) describes 55 notebook-based lessons, another separate corpus not exhaustively reviewed here.

Two relevant historical sources were verified separately:

- The [archived “Pair trading notebook” discussion](https://quantopian-archive.netlify.app/forum/threads/pair-trading-notebook.html) is practitioner anecdote, not a controlled study. One participant reports that out-of-sample cointegration was inconsistent and that the surviving, lightly traded pairs lost their apparent mean reversion when realistic quote/spread data was used. The thread also questions raw correlation as a pair-selection screen and stresses inspecting why the relationship should persist and why deviations occur. This supports testing turnover, executable spreads, and relationship breaks; it does not establish how often these failures occur or predict TheWiz results.
- The [Quantopian `research_public` pairs lecture](https://github.com/quantopian/research_public/blob/master/notebooks/lectures/Introduction_to_Pairs_Trading/notebook.ipynb) is explicitly introductory and equity-oriented. It warns that searching many pairs creates multiple-comparisons bias, cautions that z-scores can mislead under fat-tailed spreads, calls for out-of-sample checks, and shows its sample pair failing its own later cointegration check. Those lessons align with the controls above, but its dated API, equity examples, and toy strategy are not directly reusable crypto evidence.

This review covers the public material discoverable through those targeted searches. It does not amount to a complete enumeration or summary of every current community post; the community's own search/listing interface or a site-provided export would be needed to finish that inventory within its access rules.

### Paper collection and incremental scan baseline

On 2026-09-29 I followed the visible primary-source links for the highest-relevance posts and saved four SSRN PDFs through their own download buttons, plus three additional open-access crypto/pairs papers. The title-named files, source links, asset coverage, and caveats are cataloged in [`audit/quantopian_papers/papers-index.csv`](quantopian_papers/papers-index.csv); the PDFs are in [`audit/quantopian_papers/`](quantopian_papers/). The seed set for the daily scan is [`audit/quantopian_papers/community-baseline.csv`](quantopian_papers/community-baseline.csv). It is an incremental baseline for relevant posts found in the UI, **not a complete site-wide post inventory**. Future scans should keep using normal UI searches, append newly reviewed relevant URLs, and report source/access limitations rather than claiming an exhaustive crawl. SSRN's pages mark the four downloaded SSRN papers “All rights reserved. No reuse allowed without permission”; keep the PDFs for this research reference and do not redistribute them.

The “Daily Quantopian and Quant Research Paper Scan” automation remains active at 09:00 daily and has been updated to read/write this baseline and paper catalog, save accessible title-named PDFs to this folder, and report only new or materially changed findings. It is instructed to stop claiming an incremental scan if the external volume or baseline files are unavailable.

## Efficient evaluation pipeline

### 0. Freeze the experiment before viewing rankings

Record the exact hypothesis family: eligible pair universe, signal family, permitted parameters, timeframe, cost scenarios, fold boundaries, trade timing, and the primary outcome. Log every attempted pair, model, and variant, including failures and exclusions. Do not tune thresholds after seeing the final holdout.

Use a versioned run contract to bind formula/config hashes, archive receipts, universe rules, cost version, correction handling, code hash, and output location. The draft contract already binds the available config and endpoint receipt hashes; its missing approvals remain explicit in [the draft](research_run_contract_draft.json).

### 1. Cheap, deterministic data and eligibility checks

Before any statistical test, check receipt/raw hashes, closed-bar validity, duplicate and missing timestamps, venue/instrument identity, minimum history, liquidity, and funding availability. Apply only predeclared eligibility rules. Keep an exclusion ledger with a reason for every asset and pair.

Never fill or compress missing hourly bars into apparently continuous time. Fit transformations and hedge ratios using training data only. Use the point-in-time universe at each as-of time rather than the terminal list of survivors.

### 2. Use one primary baseline and a small challenger set

Start at the archive’s 1-hour frequency. Compare a small, predeclared set, for example:

- a simple distance/normalized-spread baseline;
- one explicitly specified cointegration/hedge-ratio baseline.

Avoid launching all seven strategy families, broad parameter grids, ML, and RL at once. This keeps compute and researcher degrees of freedom down. If the baseline fails after costs, more complex models should need a documented reason and a separate trial-family identifier.

Pair orientation, log/level choice, hedge-ratio estimation window, spread equation, z-score lookback/denominator, signal timestamp, next-bar execution, entry/exit, stop, maximum holding time, and re-hedging rules must be fixed before testing.

### 3. Screen, then perform inference on a declared family

Run low-cost checks on all eligible pairs, then expensive model fits on the survivors. Record the total tested family and every filter. Shared assets, common market shocks, overlapping holding periods, and repeated parameter variants make pair-test statistics dependent. Do not assume ordinary Benjamini–Hochberg FDR control is valid unless its independence or positive-dependence conditions are justified for the tested hypotheses. Preserve the policy limit of FDR <= 0.1, but choose and validate the procedure before unblinding results: Benjamini–Yekutieli is a conservative option under arbitrary dependence; a Romano–Wolf stepdown or White-style reality-check procedure can be appropriate for a predeclared family of strategy-versus-benchmark comparisons when its joint resampling assumptions fit. Use time-block resampling that preserves serial dependence and cross-asset co-movement; document the block design and sensitivity. Do not describe any procedure as controlling the project’s error rate until it is implemented and validated for this test family. The BY result is a direct consequence of the dependence conditions addressed in the original paper; Romano–Wolf’s stepwise method explicitly captures joint dependence for multiple strategy tests. [Benjamini–Yekutieli](https://doi.org/10.1214/aos/1013699998), [Romano–Wolf](https://doi.org/10.1111/1468-0262.2005.00615.x)

The family includes pairs screened out after statistical inspection and every tried signal, lookback, hedge-ratio, threshold, cost assumption, and model variant. A predeclared data-quality exclusion can remove a pair before inference; a favorable-looking performance result cannot. Use a separate untouched confirmation period for the frozen final choice.

For the selected strategy, report DSR and PBO as additional selection-risk diagnostics. Consider White’s Reality Check when comparing a broad set of models against a common benchmark. Do not stack every correction mechanically or claim any single statistic “proves” an edge. These diagnostics are complementary: multiple-testing adjustment does not repair leakage or bad cost assumptions, PBO does not replace chronological confirmation, and a high DSR cannot turn a short or stale sample into durable evidence.

### 4. Make costs part of the signal test

Calculate two-leg P&L with the actual position weights and next executable prices. Include entry and exit fees, spread/impact, funding on each leg for the actual holding interval, partial fills, and stress execution. Report turnover, capacity assumptions, and gross-to-net erosion. Compare base and stress cases before a pair is ranked as an economic candidate.

The pinned fee profile is dated 2026-08-05 and exceeds the policy's 30-day maximum; account-specific fee status is absent. L2 calibration covers BTC and ETH only. Until current, run-bound inputs exist, performance for the rest of the universe cannot be called fully after-cost or promotion-ready. The crypto studies above show why: modeled costs changed conclusions, especially for higher turnover.

### 5. Validate chronologically and at portfolio level

Use expanding or rolling chronological walk-forward folds with an embargo sized to prevent overlapping labels/holding periods and no parameter fitting on the test segment. Preserve the current five-fold/three-positive-fold requirement, but verify the fold schedule contains genuinely distinct test observations. Report fold-by-fold results, not just a pooled Sharpe. Define whether minimum trades apply per pair, per fold, or to the combined portfolio, and define how a pair with no valid trades is handled before calculation.

Then combine eligible pairs into a portfolio using a frozen sizing and rebalance rule. Test net return, drawdown, tail loss, turnover, correlated exposure, concentration, and stress costs. Apply the two-pair minimum at the portfolio level unless the owner explicitly defines another interpretation. Specify whether performance gates apply to each pair, the portfolio, or both; define trade counting, overlapping positions, leverage, and capital denominator. A pair that passes alone may add no value or may duplicate the same risk as another pair.

### 6. Require prospective confirmation before promotion

The current archive spans about seven days of hourly captures, has ten inferred missing slots, one duplicate slot, and one extra out-of-window receipt. Each capture also contains overlapping historical candle windows. Thus, 160 receipts are not 160 independent out-of-sample periods. Use this archive for data validation and preliminary screening; do not describe it as a long, independent confirmation sample.

If candidates survive retrospective tests, freeze them and observe them prospectively in shadow mode for a duration and minimum event count set before observation. Do not refresh parameters during that confirmation period. New collection would need its separate approved schedule and publication authority.

## Recommended project decision states

| State | Meaning | Typical decision |
| --- | --- | --- |
| `DATA_BLOCKED` | Hash, coverage, universe, formula, or cost prerequisites missing | Fix evidence/input gaps; no performance claim |
| `REJECT` | Fails predeclared data, statistical, risk, or stressed-cost gates | Stop this candidate; preserve the reason |
| `WATCH` | Promising exploratory result but weak/short confirmation | Freeze candidate; gather prospective shadow evidence |
| `RESEARCH_PASS` | Reproducible, after-cost, multiple-testing-controlled walk-forward evidence passes | Eligible for a separately authorized next stage, not live |
| `TESTNET_CANDIDATE` | Research pass plus venue/account compatibility and explicit order approval | Testnet only under separate authority |
| `LIVE_CANDIDATE` | Testnet evidence plus separately signed live prerequisites | Requires explicit authorization; never automatic |

## Project-specific recommendation

For TheWiz, the most useful next sequence is:

1. Register the actual Crypto Wizards UI inspection package in `/Users/gregc/TheWiz-LocalRuntime/data/research/dashboard_audit_20260915/` as a hashed, read-only historical source. Reconcile its capture index, exceptions, seven-mode screenshots, scanner-control matrix, contradiction register, and project crosswalk with the active field dictionary and capture schema. Preserve observed versus applied settings, pair orientation, venue, interval, timestamp, and source evidence; keep unknowns explicit. This is a source-mapping step, not numerical parity.
2. Keep the run contract blocked until the owner approves the method, gate mapping, correction policy, and current cost evidence. Resolve the dashboard inspection's open formula, raw-array, accounting, and asynchronous-result identity gaps before claiming exact Wizard parity.
3. Use the retained archive read-only for eligibility checks and a limited 1-hour baseline screen; preserve every attempted pair/variant in the trial ledger.
4. Preserve the policy FDR <= 0.1, but first approve a dependence-valid procedure and a declared test family; use BY or a suitable joint-resampling stepdown as candidates, not as already-selected implementation. Report DSR and PBO as additional diagnostics. Keep chronological walk-forward as the primary performance test.
5. Treat surviving pairs as `WATCH` until a frozen prospective shadow sample confirms them. The current archive is too short and gapped to establish durable performance.
6. Evaluate a combined portfolio before any separately approved Testnet stage.

This sequence is efficient because expensive modeling is reserved for candidates that pass simple predeclared data checks, while all selection steps remain auditable. It helps the project most by making a rejection or inconclusive result just as reproducible and useful as a pass.

## Self-inspection and corrections

**Self-score: 8/10 before this review; 9/10 after the corrections below.** The first draft had a real methodological omission: it named the FDR threshold as the primary gate but did not account for dependence among overlapping crypto pairs or specify what test procedure could support the claim. It also left trade-count and performance-gate aggregation less explicit than the project needs. This revision makes dependence handling an approval prerequisite and requires pair-versus-portfolio denominators to be fixed before calculation.

The remaining point is intentionally withheld because the recommended procedure has not been implemented or empirically validated on TheWiz data. This is a design audit, not evidence that a strategy passes. **Empirical strategy validation: not performed.** No pair backtest, new collection, or trading was authorized or run.
