# SKT LLM Router — what the task actually is, and where we stand

## The ceiling is 0.78, not 1.0

Every tier caps spend at a multiple of the all-light bill, and `axk1-think` costs
**23x** `ax31-light`. So even with perfect foresight only a fraction of episodes
can be upgraded. Measured on train with a perfect-information greedy fill:

| tier | budget | oracle ceiling | episodes upgraded (of 1760) |
| --- | ---: | ---: | ---: |
| fast | 1.25x | 0.7339 | 330 |
| balanced | 2.0x | 0.7886 | 447 |
| premium | 4.0x | 0.8452 | 575 |

Weighted oracle **0.7837** against an all-light **0.5973**. The whole contest is
the 0.19 in between.

## The failure that matters is the budget, not the accuracy

Scoring a tier over budget gives **zero**, and the bill uses the chosen model's
actual tokens, which are unknown when routing. A first router that ranked by
predicted efficiency and spent against predicted cost blew every tier:

```
fast 1.37x (cap 1.25)   balanced 2.69x (cap 2.0)   premium 4.89x (cap 4.0)  -> 0.0000
```

Two compounding biases. Log-space regression predicts the mean of the log, which
under-predicts the mean. And greedy **selects on low predicted cost**, so it
concentrates precisely on the episodes whose cost was under-predicted -- a
winner's curse on top of the first bias.

Predicting the **0.80 quantile of cost** in raw credits instead of the mean fixes
it. With a 0.85 budget margin every tier lands inside its cap:

| tier | score | spend | cap |
| --- | ---: | ---: | ---: |
| fast | 0.6381 | 1.04x | 1.25x |
| balanced | 0.6821 | 1.94x | 2.0x |
| premium | 0.7182 | 3.47x | 4.0x |

Weighted **0.6753** on dev, against all-light 0.6193 and a dev oracle of 0.7974
— **31.4% of the headroom**.

## What is hard: predicting the gain, not the cost

Cost predicts well (dev correlation 0.75 in log space). Gain does not:

| | gain correlation | efficiency-ranking Spearman |
| --- | ---: | ---: |
| ax31 | +0.098 | +0.018 |
| axk1-think | +0.276 | +0.168 |

For `ax31` the efficiency ranking greedy consumes is essentially random. Note the
structure that makes this tractable in principle: of the 611 train episodes where
light scores 0, `axk1-think` fixes **68.6%**; of the 954 where light scores 1, it
keeps the score **91.5%** of the time. So the task reduces to predicting *where
light fails*.

Modelling notes measured on dev, not assumed:
* char_wb 3-5 grams plus word 1-2 grams beat word features alone.
* Plain Ridge beat HistGradientBoosting on 1760 rows (0.437 vs 0.412 for light's
  own score) -- the boosting overfits at this size.
* Predicting each model's score and differencing is *worse* than regressing the
  difference directly: the shared signal cancels and only the part we need is left.

## Next

* The fast tier carries weight 0.4 and sits at 1.04x of a 1.25x cap, barely above
  all-light. That is where the weighted score is lost.
* Premium leaves 3.47x of 4.0x unspent -- the quantile is over-conservative once
  the budget is loose, so the margin should be tier-dependent, tuned on train.
* Better gain prediction is the fundamental lever, and "predict where light
  fails" is a better-conditioned target than the pairwise difference.

## Tuning the margin made it worse than doing nothing

Choosing the cost quantile and budget margin by 4-fold CV on train, for maximum
mean score, gave this on dev:

```
fast 0.6545@1.24x   balanced 0.0000@2.28x (cap 2.0)   premium 0.7202@3.66x
weighted 0.4779  -- worse than routing everything to light (0.6193)
```

Expected-score maximisation is only right when the payoff is symmetric. Here an
overrun forfeits the tier, so tuning the margin buys a probability of zero in
exchange for a few points. The margin is not a hyperparameter.

What works is structural: fill the basket using a **mid** cost quantile, then
re-price the whole basket at a **pessimistic** quantile (0.95) and drop the least
efficient purchases until even that bill fits. The first quantile decides what
looks worth buying, the second decides how wrong we can afford to be. Every tier
then lands inside its cap:

```
fast 0.6463@1.11x   balanced 0.6804@1.94x   premium 0.7071@3.05x   weighted 0.6748
```

## Do not drop ax31 for being unpredictable

Its gain correlation is +0.085 and its efficiency ranking is Spearman +0.054 --
close to random -- so removing it looks obvious. Measured:

| upgrades offered | weighted | note |
| --- | ---: | --- |
| ax31 + axk1-think | **0.6748** | |
| axk1-think only | 0.2511 | balanced and premium both overrun -> zero |
| ax31 only | 0.6668 | safe, but premium spends 1.78x of a 4.0x cap |

ax31 is not there for its accuracy, it is there because it is cheap (2.16x) and
lets the budget be filled safely. Routing only to the 23x model blows the cap.

## Embeddings help the model that matters

multilingual-e5-small (118M, 20.6% of prompts contain Hangul so an English-only
encoder is wrong):

| | tfidf | e5 | tfidf+e5 |
| --- | ---: | ---: | ---: |
| ax31 gain corr | +0.085 | +0.080 | +0.099 |
| axk1-think gain corr | +0.320 | **+0.366** | +0.329 |
| axk1-think efficiency Spearman | +0.142 | **+0.170** | +0.163 |

Caveat for the container: encoding 2,640 prompts took 138 s here, against a 105 s
budget on 2 cores. Max length must come down from 512, or the encoder has to be
smaller, before this ships.

## Where the score still is

Every configuration leaves premium at ~3.05x of a 4.0x cap. The safety pass is
over-conservative once the budget is loose, so the pessimistic quantile should
depend on the tier's slack rather than being fixed at 0.95.


## Embeddings improve the prediction and lose the competition

multilingual-e5-small lifts axk1-think gain correlation from +0.320 (TF-IDF) to
+0.361, and truncating to 96 tokens -- needed to fit 2,640 prompts into the
container's 105 s on 2 cores, since 512 tokens took 138 s -- keeps +0.345. On the
prediction metric it is clearly better.

On the actual objective it is worse:

| router | weighted dev |
| --- | ---: |
| TF-IDF only | **0.6748** |
| e5 for axk1-think, slack 0.03 | 0.6731 |
| e5, slack 0.05 | 0.6724 |
| e5, slack 0.08 | 0.6705 |

The first e5 attempt with no explicit slack landed balanced at **2.01x of a 2.0x
cap** and forfeited the tier -- weighted 0.4703. Being over by half a percent
costs everything, so the embedding router has to hold more headroom, and premium
drops from 3.05x to 2.87x of its cap. The headroom it gives up is worth more than
the prediction it gains.

Rejected, and the container keeps no encoder, no 66 s of runtime and no 2 GB
memory risk.

## Final configuration

Word 1-2 grams plus char_wb 3-5 grams, Ridge for both the gain and the log-cost
base, HistGradientBoosting quantile regression for cost. Fill on the 0.5 cost
quantile, re-price at 0.95, shed the weakest purchases until that fits, and hold
3% slack under the cap.

Dev weighted **0.6748**, all-light 0.6193, dev oracle 0.7974 -- 31% of the
headroom with every tier inside budget.


## The container path, measured honestly

`router_run.py` implements the evaluator's interface (`--input --tier --output`,
one tier per invocation) and loads a pickled artifact, since the container sees
prompts only and cannot train.

One thing the interface forces that the offline experiments hid: **the cap is a
multiple of the all-light bill, and the container is never told what that bill
is.** It has to be predicted too. The asymmetry is one-sided -- over-estimating
inflates the cap and risks an overrun that forfeits the tier, under-estimating
only leaves score unspent -- so light's own cost is fitted at a quantile chosen in
the safe direction.

Calibrating that quantile, fitted on train and evaluated on dev:

| q_light | estimated / true light bill | weighted | fast | balanced | premium |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.76x | 0.6687 | 1.05x | 1.55x | 2.67x |
| 0.50 | 0.84x | 0.6693 | 1.06x | 1.78x | 2.69x |
| 0.75 | 0.93x | **0.6719** | 1.07x | 1.85x | 2.79x |

Getting the light bill from 76% to 93% accurate moves the score by 0.003. Premium
still spends 2.79x of a 4.0x cap. **The budget is not the binding constraint any
more -- there is nothing worth buying with the rest.** With ax31's efficiency
ranking at Spearman +0.05, the router cannot tell which remaining episodes are
worth an upgrade, so unspent budget stays unspent.

That is the ceiling of this approach: honest dev weighted ~0.67, about 30% of the
oracle headroom. Everything still on the table is gain prediction, and the four
things tried today -- embeddings, cost quantiles, budget margins, model mix -- all
stopped at that same wall.

Note on a number not to trust: fitting the artifact on train+dev and scoring it on
dev gives 0.7422. That is training on the evaluation set and is reported here only
so it is not mistaken for progress.

## The 90 s limit: what the local check can and cannot tell us

`tools/check_runtime.py` fails all three tiers here, each killed at the 90 s
wall (95.004 s including the SIGTERM grace). That number is not evidence about
the official run, and the tool says so itself: it warns when the Docker server
is not the official `linux/arm64` machine.

Measured on 2,640 episodes:

| environment | wall |
| --- | ---: |
| native x86-64, 2 cores | **13.3 s** |
| QEMU-emulated arm64, same box | > 11 min (killed) |

So emulation costs a factor of roughly 50 on this workload, which is what
scalar Python plus regex-heavy vectorisation does under TCG. The official
evaluation runs native arm64 on Apple Silicon, where the honest expectation is
tens of seconds, not minutes -- but that is an expectation, not a measurement,
and nothing available here can turn it into one.

Where the time goes, natively:

| stage | wall | share |
| --- | ---: | ---: |
| TF-IDF transform (word + char_wb) | 9.6 s | 72% |
| unpickle the 9.3 MB artifact | 1.8 s | 14% |
| hand features | 1.8 s | 13% |
| all nine model predictions | 0.1 s | 1% |

The models are free; the vectoriser is the whole cost. If the native arm64 time
ever needs to come down, capping the text length fed to the char analyser is the
one lever worth pulling, and it changes predictions, so it has to be re-scored
on dev rather than assumed harmless.

## The shipped router had a 17% chance of forfeiting the balanced tier

Every configuration above was judged by its dev score and by whether each tier
landed under its cap. That is the wrong question. Dev is one sample of 880
episodes and the graded split is another, so what matters is not "did balanced
pass here" but "how far does the spend ratio move between samples of this size".

Measured, over 200 bootstrap resamples of dev, routing each resample on its own
(the ratio is a global knapsack outcome, not an average of per-episode values):

| design | balanced spend | bootstrap sd | P(over 2.0x) |
| --- | ---: | ---: | ---: |
| cost model reads the full-text n-grams (shipped) | 1.83x | 0.208 | **17.0%** |
| cost model reads hand features and log length | 1.59x | 0.106 | **0.0%** |

The margin was 0.17 against a standard deviation of 0.21. Balanced passing on
dev at 1.83x was luck, and the tier carries weight 0.3, so the honest expected
score of the shipped router is not its dev 0.6710:

| config | dev weighted | risk-adjusted |
| --- | ---: | ---: |
| full-text cost, q_safe 0.95 (shipped) | 0.6710 | 0.6362 |
| length-only cost, q_safe 0.90 | 0.6721 | **0.6721** |
| length-only cost, q_safe 0.80 | 0.6768 | 0.6141 |

The last row is the same trap a third time: the highest dev score of the three
carries a 17% chance of losing *fast*, which has weight 0.4.

Why the text hurt the cost path. Cost is close to a function of length, and the
n-grams add variance rather than signal to it -- and the greedy fill then
concentrates on whatever looks cheap in this particular sample. Removing them
costs nothing measurable on score and cuts the sampling spread in half.

## The same change fixed the runtime

Cost no longer needs the prompt, and the gain models turned out not to need all
of it: capping their input at 500 characters scored 0.6755 against 0.6748
uncapped. The tail was the whole expense -- median episode 237 characters,
longest 71,094, top 9% holding 90% of all characters.

On 2,640 episodes, native x86-64, 2 cores:

| | before | after |
| --- | ---: | ---: |
| total | 13.3 s | **4.97 s** |
| TF-IDF transform | 9.6 s | 0.7 s |
| peak RSS | | 263 MB of 2 GiB |

Two risks, one change. Nothing here was selected for score: the quantile sweep
is reported so the shape is visible, and the design was chosen on whether its
margin exceeded the measured sampling spread.

## Two attempts at the remaining headroom, both refuted

With the budget risk fixed, the open question was gain prediction and the 1.24x
of premium that goes unspent. Both were attacked and neither moved.

### "Predict where light fails" is not a better target

The structure invited it. Gain is exactly zero for 75% of episodes with `ax31`
and 61% with `axk1-think`, and what remains sits on the rows light gets wrong
(E[gain | light fails] +0.271 and +0.617, against -0.022 and -0.005 elsewhere).
So a regression on the raw difference spends its capacity on rows with no signal,
while "will light fail, and would this model fix it" is a clean 35%-positive
binary problem.

Four gain estimators, routed identically, selected on 4-fold CV *inside train*
(Spearman of the efficiency ranking the greedy consumes) and reported on dev:

| gain model | CV ax31 | CV axk1 | dev weighted | risk-adj |
| --- | ---: | ---: | ---: | ---: |
| Ridge on the difference | **+0.051** | **+0.237** | 0.6721 | **0.6721** |
| P(fail) x constant | +0.038 | +0.107 | 0.6706 | 0.6539 |
| P(fail) x conditional Ridge | +0.039 | +0.182 | 0.6735 | 0.6735 |
| calibrated P(ok) difference | +0.044 | +0.217 | 0.6753 | 0.6430 |

The rule was set before looking: adopt only what wins both. Nothing does. The
conditional Ridge is +0.0014 on dev -- far below what 880 episodes resolve -- and
loses on CV, and the calibrated difference buys its dev score with a 12% chance
of forfeiting fast. Ridge on the difference already learns where light fails,
because that is where the difference lives; splitting it into two estimates and
multiplying them only multiplies the noise.

### The unspent premium budget is the price of correlated error, not a bug

The safety pass re-prices each purchase at its own 0.9 quantile, and on premium
that sum reaches 3.85x of a 4.00x cap while the basket actually costs 2.76x. That
looks like the wrong object: the cap constrains one total, and the upper quantile
of a sum should grow like sqrt(N), not N. Replacing it with
`sum(q50) + z*sqrt(sum(sigma^2))`, sigma from the q90-q50 spread:

| bound | dev weighted | risk-adj | fast spend / P(over) | prem spend / P(over) |
| --- | ---: | ---: | --- | --- |
| sum of quantiles (current) | 0.6721 | **0.6721** | 1.11/1.25, **0%** | 2.76/4.00, 0% |
| quantile of sum, z=4 | 0.6777 | 0.5198 | 1.25/1.25, 50% | 3.24/4.00, 4% |
| quantile of sum, z=6 | 0.6748 | 0.5735 | 1.23/1.25, 38% | 2.87/4.00, 0% |
| quantile of sum, z=8 | 0.6725 | 0.6092 | 1.22/1.25, 24% | 2.73/4.00, 0% |

It spends the budget and forfeits the tiers. Even at z=8 -- eight standard
deviations, which should be unreachable -- fast overruns in a quarter of
resamples.

**Correction.** I first recorded that as evidence of correlated errors. That was
wrong, and resampling says so plainly: the sd of a sum of N residuals tracks
sqrt(N) to within 2% at N = 50, 200 and 600, so they are independent and the
sqrt is legitimate.

What was wrong is sigma. It came from `(q90 - q50)/1.2816`, the *normal* relation
between an interquantile gap and a standard deviation, and these residuals are
nowhere near normal -- skew 15.2 and kurtosis 463 for `ax31`, 8.3 and 94 for
`axk1-think`. A gap between two quantiles of a heavy-tailed distribution says
little about its spread, and here it understated the true sd by 3.07x and 2.01x,
so the bound was roughly a third of what it should have been.

Rescaling sigma by a factor measured out-of-fold on train (`ax31` x4.36,
`axk1-think` x1.99) makes the bound behave:

| bound | dev weighted | risk-adj | fast P(over) | balanced P(over) | premium spend |
| --- | ---: | ---: | ---: | ---: | ---: |
| sum of quantiles (current) | 0.6721 | **0.6721** | 0% | 0% | 2.76/4.00 |
| measured sigma, z=2.5 | 0.6732 | 0.6557 | 4% | 4% | 2.77/4.00 |
| measured sigma, z=3 | 0.6724 | 0.6704 | 0% | 1% | 2.75/4.00 |
| measured sigma, z=4 | 0.6718 | 0.6718 | 0% | 0% | 2.73/4.00 |

The corrected bound is sound and buys nothing: at a z that is actually safe it
lands exactly where the sum of quantiles already was. That is the useful result.
Premium's unspent 1.24x is not an artifact of pricing the basket wrongly -- with a
per-purchase residual sd of 208,045 credits on `axk1-think` and a kurtosis near
100, committing more of the budget genuinely cannot be done safely. The only way
to spend it is a cost model with smaller residuals.

## Where the cost residual actually is, and why a better point estimate hurts

Cost is `in_rate*in_tokens + out_rate*out_tokens`, and the three parts are not
equally hard. Input tokens correlate 0.9985 with character count and carry 2.8%
of the cost-difference variance for `axk1-think`; `num_generations` is 2 or 4 and
a logistic model on the prompt separates them at AUC 0.9990, but the two groups
have such different internal spread that knowing which is which explains about 3%
of the variance. The remaining 96% is output tokens -- how long the model thinks.

Building the cost from those parts instead of regressing the composite improves
the bulk substantially and the tail not at all:

| cost model (out-of-fold on train) | residual sd | median abs error |
| --- | ---: | ---: |
| direct, hand features + length | 194,339 | 30,973 |
| direct, plus prompt n-grams | 190,850 | 22,240 |
| composed from log tokens per model | 191,773 | **20,315** |

A 34% better median with an unchanged sd is what a kurtosis near 100 looks like.
The prediction that follows is asymmetric and worth stating before the run: the
greedy fills on the median, so score should rise, while the safety pass reads the
spread, so the spend ceiling should not move.

Half right. Score rose -- and the ceiling moved too:

| cost model | dev weighted | risk-adj | fast | balanced | premium |
| --- | ---: | ---: | --- | --- | --- |
| hand features + length | 0.6721 | **0.6721** | 1.11, 0% | 1.59, 0% | 2.76, 0% |
| plus the composed point estimate | 0.6756 | 0.6356 | 1.20, **12%** | 1.80, 4% | 3.07, 1% |

The reason is measurable. Conditioning the quantile model on an accurate point
estimate makes it narrow its intervals, and out of sample they are too narrow:

| cost model | q90 coverage, ax31 | q90 coverage, axk1-think |
| --- | ---: | ---: |
| hand features + length | 0.868 | 0.868 |
| plus the composed point estimate | 0.814 | **0.789** |

A 0.9 quantile that covers 79% is not a 0.9 quantile, and the entire safety
argument is that those quantiles are honest. This is the third time the same
shape has appeared -- a text Ridge as an auxiliary feature, then embeddings, now a
composed point estimate -- and each time the better predictor bought score with
over-confident intervals and gave back more than it took.

The generalisation worth keeping: with a heavy-tailed target and a few thousand
rows, a quantile model conditioned on a strong point estimate reports the spread
it can no longer see. Keeping the cost model deliberately weak is what makes its
uncertainty trustworthy, and trustworthy uncertainty is what the budget rule
actually consumes.
