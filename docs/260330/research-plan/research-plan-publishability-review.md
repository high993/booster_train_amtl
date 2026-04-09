# Publishability Review and Reframed Novelty

## Overall Assessment

The current plan targets a real and important problem, but the novelty claim is too broad for a 2026 submission if it is presented as "the first multi-objective automatic reward learning framework for humanoid locomotion."

As written, the project looks more like:

- a thoughtful combination of bilevel reward search
- a Pareto-style evaluation lens
- a DeepMimic-style imitation setup
- sim-to-real validation on Booster K1

That combination may still be publishable, but probably not as a strong "first-of-its-kind" algorithmic novelty claim.

The most defensible angle is narrower:

> We automatically discover interpretable and transferable reward weights for DeepMimic-style humanoid imitation, under dynamics uncertainty, and show that the resulting Pareto set reduces manual reward retuning while improving sim-to-real robustness on K1.

That is a more believable and reviewer-resistant claim.

## Recommended Reframing

### What not to claim

Avoid claiming:

- first multi-objective humanoid locomotion method
- first Pareto-based reward optimization for humanoids
- no prior work combines multi-objective control and humanoid sim-to-real

Those claims are vulnerable because recent work already covers:

- automated reward design for humanoid locomotion
- multi-objective humanoid control
- real-world humanoid deployment with competing objectives
- multi-objective robot RL in Isaac Lab-style settings

### What to claim instead

A stronger and safer framing is:

1. We focus on DeepMimic-style imitation locomotion, where reward tuning remains brittle and labor-intensive.
2. We use a bilevel outer loop to search a compact reward-weight space rather than hand-tuning task-specific weights.
3. We preserve trade-offs between tracking, energy, and smoothness by returning a Pareto set instead of a single collapsed scalar objective.
4. We evaluate whether the discovered reward weights are more transferable across motion clips and more robust under dynamics randomization.
5. We validate on a real Booster K1 platform.

This makes the contribution about:

- interpretable reward-weight discovery
- reduced per-skill retuning
- robustness under domain shift
- practical sim-to-real deployment

rather than claiming a broad conceptual first.

## Suggested Novelty Statement

Use something close to the following in the plan and later in the paper:

> This work does not claim to introduce the first multi-objective formulation for humanoid control. Instead, it studies a narrower and practically important question: can a bilevel outer loop automatically discover a compact, interpretable, and transferable reward-weight set for DeepMimic-style humanoid imitation locomotion, such that the resulting Pareto set improves transfer robustness and reduces manual reward retuning on a real K1 platform?

## Suggested Related-Work Positioning

You can use this paragraph as a starting point:

> Prior work has approached the reward-design problem for locomotion from several directions, including adversarial motion priors, automated reward generation, and multi-objective reinforcement learning. Recent humanoid studies already show that automated reward design and preference-conditioned multi-objective control can be effective in simulation and on hardware. Therefore, our contribution is not the first introduction of multi-objective control to humanoids. Instead, we focus on a different gap: automatic search over a compact and interpretable reward-weight space for DeepMimic-style imitation locomotion, with explicit attention to transfer robustness under domain randomization and reduced retuning effort on a real Booster K1.

## Is It Novel Enough?

### Short answer

Maybe, but only with the right framing and the right evidence.

### My honest read

If the final paper only shows:

- a CMA-ES or NSGA-III search over reward weights
- energy and smoothness added to an existing reward
- better curves than manual tuning in simulation

then the work will likely read as incremental.

If the paper instead shows:

- reward weights discovered automatically and reused across multiple motion clips
- a meaningful Pareto frontier with real deployment trade-offs
- reduced retuning effort versus manual reward engineering
- better robustness under dynamics randomization
- cleaner sim-to-real transfer on K1

then it becomes much more publishable.

In other words, the publishability will depend less on the optimization wrapper itself and more on whether you can demonstrate:

- transferability
- robustness
- practical reduction of engineering burden
- convincing hardware evidence

## Main Reviewer Concerns

### 1. "This looks incremental relative to recent automated reward and MORL papers."

This will be the first concern. If reviewers see the method as "multi-objective search over reward weights," they may view it as a combination paper rather than a new method.

How to respond:

- avoid first-of-kind claims
- emphasize the DeepMimic imitation setting
- emphasize interpretable weight discovery
- emphasize sim-to-real robustness and reduced retuning burden

### 2. "Why is this bilevel rather than just black-box hyperparameter search?"

If the outer loop is implemented with ES or CMA-ES over reward weights, reviewers may say the method is simply expensive search over hyperparameters.

How to respond:

- define the bilevel problem formally
- explain clearly what the inner optimization solves
- show why direct manual tuning or scalar search is insufficient
- demonstrate that the outer-loop objective is tied to transfer or robustness, not only training return

### 3. "The Pareto story is weak if you eventually choose one operating point."

If you only generate a Pareto frontier and then pick one final reward, reviewers may ask why scalarization was not enough.

How to respond:

- show multiple useful deployment choices on the frontier
- show that different skills or hardware settings prefer different Pareto points
- quantify frontier quality, not just final-point performance

### 4. "Energy and smoothness are standard regularizers, not a novel contribution."

Adding common reward terms alone will not count as novelty.

How to respond:

- do not present the reward terms themselves as novel
- present the automatic trade-off discovery and transfer behavior as the contribution

### 5. "Your abbreviated PPO evaluation may not predict final performance."

This is a serious technical concern. If short inner-loop training is used for the outer-loop fitness, reviewers will question whether it is a valid proxy.

How to respond:

- run a correlation study between abbreviated evaluation and full training outcomes
- report rank correlation, not only anecdotal agreement
- show failure cases and explain limits

### 6. "The gains may come from domain randomization rather than reward search."

If domain randomization and reward optimization are introduced together, reviewers may not know what caused the improvement.

How to respond:

- include clean ablations:
- manual reward without domain randomization
- manual reward with domain randomization
- learned reward without domain randomization
- learned reward with domain randomization

### 7. "The baselines are not strong enough."

AMP, Eureka, and a scalar bilevel baseline are probably not enough for a 2026 paper.

How to respond:

- add at least one recent multi-objective humanoid or robot RL baseline
- include a simple but strong scalarized search baseline
- include a low-cost tuning baseline so the compute trade-off is visible

### 8. "The scope is too narrow: one robot, two skills."

A single K1 with only walking and running may feel limited unless the paper demonstrates broader reuse.

How to respond:

- test multiple motion clips
- test perturbations or dynamics shifts
- show reuse of discovered weights across tasks or environments

### 9. "The hardware evidence is too weak."

A few successful videos or isolated rollouts will not be convincing.

How to respond:

- report repeated trials
- report success rate and fall statistics
- report performance variance
- report transfer gap explicitly

## What Would Make the Paper Stronger

### Highest-impact upgrades

1. Reframe the paper around transfer-robust reward discovery, not generic MORL novelty.
2. Add a cross-skill reuse result: one discovered reward configuration or Pareto set reused across several motion clips.
3. Validate abbreviated PPO as an outer-loop proxy with rank-correlation analysis.
4. Add stronger recent baselines from humanoid or robot multi-objective RL.
5. Separate reward-search gains from domain-randomization gains through ablations.

### Nice-to-have upgrades

1. Show that the learned reward weights are interpretable and stable across seeds.
2. Show that manual tuning time is reduced in practice, not just in principle.
3. Report compute cost honestly and compare it to human tuning effort.

## Suggested Revised Paper Pitch

If you want a one-paragraph pitch for the project, I would use this:

> We present a sim-to-real robust reward-search framework for DeepMimic-style humanoid imitation on the Booster K1. The method uses a bilevel outer loop to automatically discover interpretable reward weights while preserving trade-offs between tracking fidelity, energy use, and motion smoothness through Pareto-based evaluation. Rather than claiming a first multi-objective humanoid controller, the paper asks whether automatic reward-weight discovery can reduce manual retuning and improve transfer robustness across skills and randomized dynamics. We evaluate the method in simulation and on real K1 hardware against manual tuning, scalarized reward search, and recent reward-design baselines.

## Final Recommendation

My recommendation is:

- keep the project
- narrow the novelty claim
- strengthen the baselines
- make transferability and reduced retuning the central thesis

With those changes, the idea looks plausibly publishable.

Without those changes, it is at real risk of being judged as an incremental combination of known ingredients.
