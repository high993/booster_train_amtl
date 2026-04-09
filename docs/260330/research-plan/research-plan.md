# Research Plan: Automated Multi-Objective Reward Learning for Humanoid Locomotion (CoRL 2026)

## TL;DR

The paper addresses the reward engineering bottleneck in humanoid robot RL by proposing a **Bilevel Multi-Objective Reward Optimization (BiMO-Reward)** framework that automatically discovers reward function weights for the Booster K1 humanoid. The key novelty is combining bilevel optimization (outer loop learns reward weights, inner loop trains policy) with Pareto-based multi-objective optimization to avoid scalar collapse of conflicting objectives (tracking accuracy vs. energy efficiency vs. motion naturalness). Validated on walking and running tasks with sim-to-real transfer to the physical K1.

---

## 1. Problem Statement

### Core Issue
Reward function design for humanoid robot locomotion remains a **manual, brittle, and non-transferable** process. The current DeepMimic-style approach requires:

- **5 hand-tuned weight parameters** (`reward_pose_w`, `reward_vel_w`, `reward_root_pose_w`, `reward_root_vel_w`, `reward_key_pos_w`) — see `ppo_config.yaml` lines 56–60
- **5 hand-tuned scale parameters** (`reward_pose_scale`, `reward_vel_scale`, etc.) — see `ppo_config.yaml` lines 61–65
- **Phase-specific multipliers** for critical moments (landing phase) — hardcoded in `deepmimic_env.py`
- **Asymmetric scale ranges** (0.01 to 12.0) making manual tuning extremely fragile

Each new motion skill, robot morphology change, or sim-to-real transfer requires re-tuning these 10+ parameters. Adversarial approaches (AMP/ASE) reduce this but introduce discriminator pre-training overhead and mode collapse risks.

### What Exists vs. What's Missing
- **Current bilevel in codebase** (`bilevel_reward.py`): Simple two-timescale weight adaptation using deficit-based heuristic. Not true bilevel optimization — no gradient-based outer loop, no principled multi-objective handling.
- **Reference paper** (Automatic Reward Learning): Uses bilevel optimization but collapses objectives into a scalar, losing the Pareto structure.
- **Gap**: No existing work combines bilevel reward optimization with explicit multi-objective Pareto optimization for humanoid locomotion with sim-to-real validation.

---

## 2. Proposed Framework: BiMO-Reward

### Architecture Overview

```
Outer Loop (Reward Optimization):
  ├── Pluggable MOO Optimizer  ←─────────────────────────────────────┐
  │   ├── [default]  NSGA-III  (population-based, Pareto frontier)   │
  │   ├── [alt]      MOEA/D    (decomposition, preference vector λ)   │  swap via
  │   ├── [alt]      AMTL-Median (task-loss median aggregation)       │  config flag
  │   ├── [alt]      STCH      (smooth Tchebycheff scalarization)     │
  │   ├── [alt]      UPGrad    (gradient-based, needs diff. inner)    │
  │   └── [alt]      Scalar Bilevel (ablation: no Pareto)             │
  │                                                                   ┘
  │   Objectives evaluated on held-out rollouts (NOT used as reward):
  │   ├── f₁: Tracking accuracy  — mean tracking_lin_vel + tracking_ang_vel
  │   ├── f₂: Energy efficiency  — Cost of Transport (CoT)
  │   └── f₃: Motion naturalness — mean joint acceleration ‖q̈‖²
  │
  │   Pareto frontier → Select operating point via preference vector λ
  │
  └── Output: w* = [w_tracking, w_energy, w_smooth]  (scales reward terms below)

Inner Loop (Policy Training):
  └── PPO with reward = w_tracking·r_track + w_energy·r_power + w_smooth·r_acc
      → π*(w*)   [black box from outer loop's perspective]
```

> **Key design principle:** The outer-loop optimizer is an abstract interface (`MooOuterOptimizer`). All six optimizer variants above implement the same `ask() / tell() / pareto_front()` API. Switching between them requires only a config flag — the inner loop (PPO), reward evaluator, and fitness functions are untouched. This enables seamless ablations and comparisons across optimizer families.

### Key Components to Build

**A. Pluggable MOO Optimizer Interface (`utils/moo_optimizer.py`)**
```python
class MooOuterOptimizer(ABC):
    def ask(self) -> list[np.ndarray]:          # propose weight vectors to evaluate
    def tell(self, weights, objectives):         # receive (w, F(w)) feedback
    def pareto_front(self) -> list[np.ndarray]: # return current non-dominated set
```
Concrete implementations (all drop-in replaceable):

| Class | Method | Inner Loop Requirement |
|---|---|---|
| `NsgaIIIOptimizer` | NSGA-III (`pymoo`) | Black-box PPO ✓ |
| `MoeadOptimizer` | MOEA/D (`pymoo`) | Black-box PPO ✓ |
| `AmtlMedianOptimizer` | AMTL-Median loss aggregation | Black-box PPO ✓ |
| `StchOptimizer` | Smooth Tchebycheff scalarization | Black-box PPO ✓ |
| `UpGradOptimizer` | UPGrad conflicting-gradient descent | Differentiable inner loop (MAML-PPO) |
| `ScalarBilevelOptimizer` | CMA-ES + scalar collapse | Black-box PPO ✓ (ablation) |

**B. Reward Evaluator — Separation of Reward Terms and Evaluation Metrics**
- The inner PPO policy is trained with: $r = w_{track}\cdot r_{track} + w_{energy}\cdot r_{power} + w_{smooth}\cdot r_{acc}$
- The outer loop evaluates **independent fitness functions** on held-out rollouts:
  - $f_1$: mean `tracking_lin_vel_x/y` + `tracking_ang_vel` (from `_reward_tracking_*`)
  - $f_2$: **Cost of Transport** $\text{CoT} = \frac{\sum_t\sum_j|\tau_j\dot{q}_j|\cdot\Delta t}{m\,g\,d}$ (from `self.torques`, `self.dof_vel`, root displacement)
  - $f_3$: mean $\|\ddot{q}\|^2$ (from `_reward_dof_acc`: `(last_dof_vel - dof_vel)/dt`)
- **Critical:** $f_2$ (CoT) ≠ $r_{power}$ (reward term). Evaluation and reward shaping are deliberately decoupled to prevent reward-hacking of the metric.
- Location: new `utils/reward_evaluator.py`

**C. Energy Cost — Three Evaluation Levels**
1. **Sim (inner loop):** `_reward_power = Σ(τ·q̇).clip(min=0)` — positive mechanical work only, available every step via `self.torques` and `self.dof_vel` (already implemented in `envs/t1.py` L750)
2. **Sim (outer loop / paper metric):** CoT over full episode — dimensionless, robot-size-independent, comparable across baselines
3. **Real hardware:** battery current × voltage integrated over walking trial — ground truth; ratio sim-CoT / real-CoT is itself a reported sim-to-real metric

**D. Sim-to-Real Transfer Module**
- Domain randomization integration in Isaac Lab engine
- Reward robustness evaluation across randomized dynamics
- Deployment pipeline for K1 MuJoCo → K1 real hardware

---

## 3. Evaluation Metrics

### Primary Metrics (for paper results)

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Tracking Error** | $\frac{1}{T}\sum_t \|q_t - q_t^{ref}\|^2$ | Mean squared joint angle error over episode |
| **Root Tracking Error** | $\frac{1}{T}\sum_t (\|p_{root,t} - p_{root,t}^{ref}\|^2 + \|R_{root,t} \ominus R_{root,t}^{ref}\|^2)$ | Root position + orientation error |
| **Energy Efficiency (CoT)** | $\text{CoT} = \frac{\sum_t\sum_j|\tau_j\dot{q}_j|\Delta t}{m\,g\,d}$ — dimensionless, robot-size-independent | Computed from `self.torques` × `self.dof_vel`, root displacement; **separate from** `_reward_power` used in training |
| **Motion Smoothness** | $\frac{1}{T}\sum_t \|\ddot{q}_t\|^2$ | Joint acceleration magnitude |
| **Reward Tuning Time** | Wall-clock time to reach convergent reward weights | Hours on single GPU |
| **Hypervolume Indicator** | Volume dominated by Pareto frontier | Standard MOO metric |
| **Sample Efficiency** | Training steps to reach 80% of converged performance | Inner-loop sample count |

### Secondary Metrics (sim-to-real)

| Metric | Description |
|--------|-------------|
| **Sim-to-Real Gap** | Performance drop when transferring to physical K1 |
| **Real-World Walking Distance** | Distance walked before falling |
| **Zero-Shot Transfer Success Rate** | % of trials succeeding without real-world fine-tuning |

### Baseline Comparison Matrix

| Baseline | Description | What it tests | Outer loop type |
|----------|-------------|---------------|----------------|
| **Manual DeepMimic** | Current hand-tuned 5-weight reward (`ppo_config.yaml` defaults) | Automation benefit | None (manual) |
| **AMP** | Adversarial Motion Prior discriminator reward | Implicit vs. explicit objectives† | None (discriminator) |
| **Scalar Bilevel** | CMA-ES bilevel with scalar-collapsed objective | MOO benefit | Single-objective ES |
| **AMTL-Median** | Adaptive multi-task learning with median loss aggregation | Robust aggregation vs. Pareto | Gradient-based MTL |
| **STCH** | Smooth Tchebycheff scalarization outer loop | Preference-conditioned scalarization vs. Pareto | Evolutionary / gradient |
| **UPGrad** | Conflicting-gradient MOO (requires differentiable inner) | Gradient-based MOO vs. population-based | Gradient-based |
| **Eureka (LLM)** | GPT-4 generated reward functions | Learning- vs. LLM-based | LLM search |
| **BiMO-Reward (ours)** | NSGA-III bilevel + explicit Pareto frontier | Full framework | Population-based MOO |

†AMP automates reward *specification* (what natural looks like) via a discriminator; BiMO-Reward automates reward *weighting* (how much each explicit objective matters) via MOO. See Section 4.5 for philosophical distinction.

---

## 4. Areas of Focus and Framework Choices

### 4.1 Bilevel Optimization Method

**Recommended approach: Evolutionary Strategies (ES) outer loop**
- Why: Non-differentiable reward-to-policy mapping (PPO is not end-to-end differentiable). ES avoids need for implicit differentiation.
- Alternative considered: Implicit differentiation (requires differentiable inner loop approximation — more complex, higher risk for deadline)
- Implementation: Replace `AutoRewardLearner.update()` with CMA-ES or OpenAI-ES operating on the weight vector

### 4.2 Multi-Objective Optimization Method — Pluggable Interface

All outer-loop optimizers implement a common `MooOuterOptimizer` interface (see Section 2, Component A), enabling seamless switching. The inner loop (PPO), reward evaluator, and fitness functions remain unchanged across all variants.

**Default: NSGA-III** (recommended for paper primary results)
- Population of reward weight vectors (~20), each evaluated by abbreviated 200-epoch PPO
- Pareto frontier yields a set of non-dominated reward configurations
- Final selection via preference vector λ (user specifies: tracking vs. efficiency vs. smoothness)
- Library: `pymoo`; gradient-free, suited to non-differentiable PPO inner loop

**Alternative outer-loop optimizers (for ablation/comparison):**

| Optimizer | Philosophy | Gradient req. | Pareto coverage | Best for |
|---|---|---|---|---|
| **NSGA-III** | Reference-point dominated sorting | None (black-box) | Full frontier | Default |
| **MOEA/D** | Decompose into scalar subproblems via λ | None | Dense on λ directions | When preference direction known |
| **AMTL-Median** | Aggregate task losses via weighted median (robust to outlier objectives) | None | Single point per λ | Robustness to noisy fitness evaluations |
| **STCH** | Smooth Tchebycheff: $\min_w \max_i \lambda_i(f_i - z_i^*)$ with smoothing | None | Single point per λ | Preference-conditioned, convex Pareto |
| **UPGrad** | Resolve conflicting gradients; project to common ascent direction | $\nabla_w F(w)$ needed | Single Pareto-stationary point | Only viable with MAML-PPO inner loop |
| **Scalar Bilevel** | CMA-ES on $\sum_i \lambda_i f_i$ | None | None (scalar collapse) | Ablation: quantify MOO benefit |

**Note on UPGrad:** Requires $\nabla_w F(w)$, i.e. gradient of multi-objective fitness w.r.t. reward weights. Not available with standard PPO. Viable only if inner loop is replaced with MAML-style few-step differentiable policy update. Include as stretch contribution if MAML-PPO is implemented; otherwise include as theoretical discussion.

### 4.3 What to Adjust in the Codebase

**Files to modify:**
- `ppo_bundle/bilevel_reward.py` — Replace `AutoRewardLearner` with ES-MOO outer loop
- `envs/env.py::compute_reward` — Add energy and smoothness reward terms
- `ppo_bundle/deepmimic_ppo_env.py::_compute_reward_terms` — Mirror new terms
- `ppo_bundle/trainppo.py` — Add outer-loop orchestration, abbreviated training for fitness evaluation
- `ppo_config.yaml` — Add new hyperparameters for outer loop
- `envs/deepmimic_env.py` — Remove hardcoded landing phase multipliers (let optimizer discover these)

**Files to create:**
- `utils/moo_optimizer.py` — Abstract `MooOuterOptimizer` base class + all 6 concrete implementations (NSGA-III, MOEA/D, AMTL-Median, STCH, UPGrad, ScalarBilevel)
- `utils/reward_evaluator.py` — Abbreviated PPO training + multi-objective fitness evaluation; computes CoT separately from `_reward_power`
- `scripts/run_bilevel_moo.py` — Top-level experiment script; `--optimizer` flag selects outer-loop algorithm
- `scripts/sim2real_eval.py` — Sim-to-real evaluation pipeline

### 4.5 BiMO-Reward vs. AMP: Philosophical Distinction

This distinction is critical for positioning the paper's contribution relative to AMP/ASE:

| Dimension | AMP | BiMO-Reward |
|---|---|---|
| **What is automated** | *Specification* — what natural motion looks like (via discriminator trained on mocap) | *Weighting/balancing* — how much each explicit objective matters (via bilevel MOO) |
| **Objective representation** | Implicit: black-box discriminator score | Explicit: interpretable mathematical terms (CoT, joint acceleration, tracking error) |
| **Multi-objective treatment** | Collapses to scalar: $w_{task}\cdot r_{task} + w_{style}\cdot r_{style}$ | Maintains full Pareto frontier; no scalar collapse |
| **Data requirement** | Mocap clips to train discriminator | Reference clips for tracking only; energy & smoothness are analytical |
| **Interpretability** | "Does it look like mocap?" — not decomposable | Each weight $w_i$ directly controls a quantifiable objective |
| **User control** | One knob: $w_{style}$ vs $w_{task}$ (still hand-tuned) | Continuous navigation of Pareto frontier via preference vector λ |
| **New skill transfer** | Retrain discriminator on new clips | Rerun Pareto outer loop; objective functions unchanged |

**Paper positioning:** AMP and BiMO-Reward solve *different parts* of the reward engineering problem and are therefore complementary. A future extension could use an AMP discriminator as one of the BiMO-Reward objectives — treating "AMP style score" as $f_3$ instead of joint acceleration smoothness. This is noted as future work.

### 4.6 Sim-to-Real Strategy

- Isaac Lab domain randomization (friction, mass, motor delay, sensor noise) — extend `engines/isaac_lab_engine.py`
- Outer loop evaluates reward robustness across randomized dynamics (reward weights that work across perturbations are preferred)
- Transfer to K1 via ROS2 bridge or direct motor command interface
- Real-hardware energy measurement: battery current × voltage integrated over walking trial (ground truth CoT); report sim-CoT / real-CoT ratio as sim-to-real energy fidelity metric

---

## 5. Implementation Phases

### Phase 1: Foundation (Week 1-2) — *Parallel tasks*
1. **Extend reward function** — Add energy efficiency and smoothness terms to `compute_reward` in `envs/env.py` and `deepmimic_ppo_env.py`
2. **Implement ES-based outer loop** — Replace `AutoRewardLearner` with CMA-ES, using `pycma` library. Each candidate = weight vector, fitness = abbreviated 200-iteration PPO run
3. **Set up MOO framework** — Integrate `pymoo` NSGA-III with the ES evaluations
4. **Baseline: AMP integration** — Complete the missing discriminator training in `trainppo.py` using the existing `AMPEnv` infrastructure

### Phase 2: Core Experiments (Week 3-4) — *Depends on Phase 1*
5. **Walking task experiments** — Train BiMO-Reward and all baselines on walking motion clips
6. **Running task experiments** — Same on running motion clips  
7. **Ablation studies:**
   - BiMO (full) vs. Bilevel-only (no MOO) vs. MOO-only (no bilevel) vs. Manual
   - Sensitivity to outer-loop population size, abbreviated training length
   - Pareto frontier visualization

### Phase 3: Sim-to-Real (Week 5-6) — *Depends on Phase 2*
8. **Domain randomization sweep** — Integrate dynamics randomization into outer-loop evaluation
9. **K1 deployment** — Transfer best Pareto-selected policy to physical K1
10. **Real-world evaluation** — Walking/running distance, stability, energy consumption

### Phase 4: Paper Writing (Week 6-8) — *Parallel with Phase 3*
11. **Results compilation** — Tables, Pareto frontier plots, learning curves
12. **Paper draft** — Introduction, Related Work, Method, Experiments, Conclusion
13. **Review and polish** — Internal review, camera-ready

---

## 6. Literature to Read

### Must-Read (Core)

1. **Peng et al. (2018)** — "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills" — Foundation of the current codebase
2. **Peng et al. (2021)** — "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation" — Baseline method
3. **Peng et al. (2022)** — "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" — Extension of AMP
4. **Reference paper in repo** — "Deep Reinforcement Learning for Real-World Humanoid Robot Locomotion Control with Automatic Reward Learning" — Direct predecessor, bilevel approach
5. **Ma et al. (2023, Eureka)** — "Eureka: Human-Level Reward Design via Coding Large Language Models" — LLM baseline
6. **Deb et al. (2014)** — "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach (NSGA-III)" — MOO algorithm

### Should-Read (Methods & Theory)

7. **Berducci et al.** — "Hierarchical Potential-Based Reward Shaping (HPRS)" — Referenced in bilevel-moo-integration notes, hierarchical reward ordering
8. **Jackson and Channon** — Relative weighting and temporal structure in humanoid gaits — Demonstrates sensitivity of reward weighting
9. **Lu (regret minimization bilevel)** — Referenced in bilevel-moo-integration notes, theoretical bilevel framework
10. **Hansen (2016)** — "The CMA Evolution Strategy: A Tutorial" — Outer-loop optimization method
11. **Zhang & Li (2007)** — "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition" — MOEA/D outer-loop alternative
12. **Xu et al. (2020)** — "Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control" — MORL for robotics
13. **Liu et al. (2021)** — "Conflict-Averse Gradient Descent for Multi-task Learning" (CAGrad / UPGrad family) — Gradient-based conflicting-gradient resolution; basis for `UpGradOptimizer`
14. **Navon et al. (2022)** — "Multi-Task Learning as a Bargaining Game" (Nash-MTL) — Related gradient-based MTL method; contextualizes AMTL-Median
15. **Momma et al. (2022)** — "A Multi-objective / Multi-task Learning Framework Induced by Pareto Stationarity" (STCH) — Smooth Tchebycheff scalarization; basis for `StchOptimizer`
16. **Lin et al. (2019)** — "Pareto Multi-Task Learning" — Theoretical basis for Pareto-stationary solutions in MTL; connects MOO and multi-task RL

### Good-to-Read (Sim-to-Real & Practice)

13. **Radosavovic et al. (2024)** — "Real-World Humanoid Locomotion with Reinforcement Learning" — Sim-to-real for humanoids
14. **Kumar et al. (2021)** — "RMA: Rapid Motor Adaptation for Legged Robots" — Sim-to-real adaptation
15. **Rudin et al. (2022)** — "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning" — Isaac Gym training at scale
16. **Smith et al. (2023)** — "Grow Your Limits: Continuous Improvement with Real-World RL for Robotic Locomotion" — Real-world RL fine-tuning
17. **Li et al. (2024)** — "Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control" — Recent bipedal RL survey

### K1-Specific & Hardware

18. **Booster K1 technical documentation** — Joint limits, torque limits, sensor specs (from `K1_serial.xml`: 26 DOF, torque ranges 7-10 Nm for arms, leg torques TBD from actuator specs)

---

## 7. Decisions & Scope

### Included
- Bilevel + MOO framework for automatic reward weight discovery
- Pluggable outer-loop optimizer interface (NSGA-III default; MOEA/D, AMTL-Median, STCH, Scalar Bilevel as ablations; UPGrad as stretch)
- Walking and running locomotion tasks
- Sim-to-real transfer to Booster K1
- Comparison against 7 baselines/ablations (Manual DeepMimic, AMP, Scalar Bilevel, AMTL-Median, STCH, UPGrad, Eureka)
- Three-level energy evaluation: per-step reward signal, episodic CoT (paper metric), real-hardware battery measurement

### Excluded
- Backflip and other acrobatic motions (too risky for sim-to-real in timeline)
- Full AMP discriminator training from scratch (use as baseline only, not as component of our method)
- Reward *structure* discovery (we optimize weights, not the functional form of reward terms)
- Multi-robot generalization (K1 only)

### Key Assumptions
- Isaac Lab can provide torque and acceleration data for energy/smoothness rewards (verified: available via engine API)
- Abbreviated PPO training (200 iters) is a reliable proxy for full training convergence direction
- K1 hardware will be available for real-world experiments by Week 5

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Outer loop too expensive (N_pop × 200 PPO iters) | Use surrogate model after initial evaluations; parallelise on multi-GPU |
| MOO doesn't converge in time | Fall back to weighted Chebyshev scalarization (still better than naive sum) |
| Sim-to-real gap too large | Add domain randomization to outer loop; report sim-only results as valid contribution |
| AMP baseline hard to implement | Use published AMP results or simplified GAN discriminator |
| Eureka baseline requires GPT-4 API | Use published Eureka reward templates, adapt to K1 |

---

## 9. Suggested Paper Structure

1. **Introduction** — Reward engineering bottleneck, K1 platform, contribution summary
2. **Related Work** — DeepMimic, AMP/ASE, bilevel optimization, MORL, LLM reward design
3. **Background** — PPO, bilevel optimization formulation, MOO preliminaries
4. **Method: BiMO-Reward** — Framework, outer-loop MOO, inner-loop policy training, Pareto selection
5. **Experimental Setup** — K1 model, Isaac Lab, tasks, baselines, metrics
6. **Results** — Quantitative comparison, Pareto frontiers, ablations, sim-to-real
7. **Discussion** — Limitations, computational cost, generalization
8. **Conclusion**
