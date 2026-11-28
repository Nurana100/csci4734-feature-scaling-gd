# Feature Scaling as Preconditioning for Gradient Descent

> How do standardization, min-max scaling and whitening change the Hessian and the convergence of gradient descent for linear and logistic regression, and what does each one cost?

**Team:** Nurana Aliyeva, Nijat Alaskarli, Ilknur Huseynova • **Course:** CSCI4734 Machine Learning, ADA University (Fall 2026) • **Topic:** 3, Feature scaling as preconditioning

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/Nurana100/csci4734-feature-scaling-gd.git
cd csci4734-feature-scaling-gd
pip install -r requirements.txt

# 2. Check that the code works
python src/demo.py

# 3. Run the experiments (from the repo folder, results are saved to results/)
python src/experiment_e1.py   # steps vs feature scale ratio
python src/experiment_e2.py   # steps vs feature correlation

# 4. Make the figures for E1 and E2
python src/plot_e1_e2.py
```

## Idea

Gradient descent is slow when features have very different scales or are correlated, because the loss surface becomes a long narrow valley. The number of steps grows in proportion to the condition number of the Hessian. Scaling the features changes the Hessian, so we measure how many steps each scaling method needs and how much it costs to compute. The fairness detail: the L2 penalty is transformed with the scaling, so every method solves the same optimization problem.

## Headline result

Linear regression, scale ratio 100, feature correlation 0.5, stopping at objective gap below 1e-8:

| Scaler | Hessian condition number | Steps |
|---|---|---|
| None | 499,281 | did not converge in 20,000 |
| Min-max | 426 | 3,912 |
| Standardization | 12.9 | 114 |
| Whitening | 1.0 | 3 |

## Figures

![E1: steps vs feature scale ratio](results/e1_iterations.png)

![E2: steps vs feature correlation](results/e2_iterations.png)

![Steps vs condition number](results/iterations_vs_condition_number.png)

## Experiments

| ID | What it measures | Owner | Status |
|---|---|---|---|
| E1 | Steps to converge vs feature scale ratio | Nurana | done |
| E2 | Steps to converge vs feature correlation | Nurana | done |
| E3 | Learning-rate sensitivity | Nijat | in progress |
| E4 | Total cost: scaling time plus gradient descent time | Ilknur | in progress |
| E5 | L2 penalty equivalent across scalers | Nijat | in progress |

Derivations: D1 and D2 (Nurana, `report/D1_D2.txt`), D4 (Nijat), D5 (Ilknur).

## Repository structure

```
src/         code: data_gen.py, scalers.py, gd.py, experiment_e1.py, experiment_e2.py, plot_e1_e2.py, demo.py
results/     CSV files, figures and short notes for each experiment
report/      report drafts and derivations
slides/      presentation
notebooks/   demo notebook
```

## Working in the repo

1. Run `git pull` before you start.
2. Commit your own files so your name shows in the history: `git add .`, `git commit -m "short message"`, `git push`.
3. If the push is rejected, run `git pull`, then `git push` again.
4. Do not edit `data_gen.py`, `scalers.py` or `gd.py` directly. Tell Nurana what to change.

## Deadlines

Results freeze: Dec 4 • Submission: Dec 14, 23:59
