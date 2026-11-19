# strata-starter

Starter code for the C1 Research Computing and Software Development coursework,
University of Cambridge, Michaelmas 2026. The coursework brief gives the questions
and formulas.

## Installation

Use Python 3.12 with the supplied dependency files. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -c "import jax; print(jax.devices())"
```

On the course TPU VM, replace the `requirements.txt` installation with:

```bash
python -m pip install -r requirements-tpu.lock -r requirements-course.txt
```

Both setups include Matplotlib, pytest and the submission checker. Python 3.12
must be installed before creating the environment.

## Supplied code

| Module or function | Purpose |
|---|---|
| `strata.sim.cohort(seed, n, p)` | Generate the cohort; `geno` contains the genotype matrix and `pheno` the trait vector. |
| `strata.reference` | NumPy reference implementations, using float64 by default. |
| `strata.angles.principal_angles(Q1, Q2)` | Compare two subspaces; returns angles in degrees, in ascending order. |
| `strata.bench.timeit(f, *args)` | Time execution after the first call and wait for each computation to finish. |

Write the five JAX functions in `src/strata/kernels.py` and add the tests requested
in Question 1. The supplied NumPy implementations are references for those tests.

## Submitting answers

**Deadline: Wednesday 16 December 2026 at 23:59 (UK time).**

Submit your work to **your own private GitLab project** at:

```text
https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/c1_coursework/<YourCRSid>
```

Replace `<YourCRSid>` with your CRSid. Only you and authorised course staff
will have access to this project. Your access expires at the deadline.
Do not grant other students, including teammates, access to your submission project.
This public repository is for distributing the starter. Do not submit answers
through its issues or pull requests.

Each student submits:

- `report/report.pdf`: an individual report of at most **3000 words**,
  organised by Questions 1–4 and their parts, with explanations, comparisons
  and labelled plots.
- `results/results.json`: the numerical results and paths to the report,
  code, test log and plots, using the supplied format.

Complete Questions 1 and 2 individually and tag their commits before sharing
code for the team work. For Questions 3 and 4, include the team's code,
measurements and plots in your own project. Each student writes their own
report and completes their own results file. Share only the material permitted
for the team questions; keep individual reports and Questions 1 and 2 private.
The final `<crsid>-q4` tag must contain your complete submission; the earlier
tags preserve the work for each question.

The word limit is an upper limit; a shorter report can receive full marks.
Question 4(b) retains its limit of 150 words within the report. Use selected tables
where they help the discussion; the JSON records the complete numerical results.
The two plots must appear in the report and also be saved as separate files.

The LaTeX report template is optional; any editor that exports PDF is suitable.
For LaTeX, edit the template and compile it from the project root:

```bash
pdflatex -output-directory=report report/report.tex
```

Replace every `null` in the JSON and keep its keys and case identifiers unchanged.
Keep `report_path` set to `report/report.pdf`. Enter your CRSid in `student.crsid`.

| Question | JSON section | Results to record |
|---|---|---|
| 1(a) | `q1.a` | Path to the JAX implementation. |
| 1(b) | `q1.b` | Test files, execution log and measured errors. |
| 2(a) | `q2.a` | Eleven coefficient values; explain the counts in the report. |
| 2(b) | `q2.b` | Predictions for the eleven specified kernel/size combinations. |
| 2(c-e) | `q2.c`, `q2.d`, `q2.e` | Scaling exponents, batch-size threshold and memory-capacity limit. |
| 3(a) | `q3.a` | Three GRM measurements, buffer sizes, fitted exponent and host-CPU comparison. |
| 3(b) | `q3.b` | Eigenvector timings, their ratio and the two angles. |
| 3(c) | `q3.c` | Six scan measurements, the first tested crossing and the scan plot. |
| 4(a) | `q4.a` | Path to the roofline plot. |
| 4(b) | Report only | Interpretation, at most 150 words. |

`q3.environment` records the TPU device descriptions and JAX version. Questions
1 and 2 are individual work. Team members may share Question 3 measurements and
plots; each student writes their own report, including all explanations and the
Question 4(b) interpretation.

Use the supplied JSON template, format version 2.1.

If you started with version 2.0, move your completed report to `report/report.pdf`
and your completed JSON to `results/results.json`. Set `schema_version` to `2.1`
and `report_path` to `report/report.pdf`. The numerical fields are unchanged.
Preserve your completed answers when updating the template, and commit all
referenced files in your own project.

### Units and files

- Times are in **seconds**, rates in **FLOP/s**, and memory sizes in **bytes**.
- `percent_peak` is a percentage; `roof_time_over_time` and `speedup` are ratios.
- Enter JSON numbers, not strings containing units. `1.2e-3` is valid JSON.
- Store plots as PDF or PNG. Paths are relative to the project root, using `/`.
  Commit every referenced file, including the test log.
- The two Question 1 angles come from the full `n × k` outputs. Question 3(b)
  compares the first two columns, as specified in the brief.

The schema in `results.schema.json` describes each field. To save results from
Python, use `json.dump(results, file, indent=2, allow_nan=False)`; convert NumPy or
JAX scalars with `float(...)` or `int(...)`, and arrays with `.tolist()`.

### Validation

Run from your project root to check your submission:

```bash
python check_submission.py
```

The checker follows the same approach as the S1 `check_submission.py`: it checks
the README, opens the report PDF, validates the JSON format and checks referenced
files. It prints `PASS` or `FAIL` and exits with a non-zero status on failure.
It does not assess numerical correctness, report content or word counts, or
execute submitted code.

During preparation, `--draft` permits unanswered `null` fields and skips checking
whether the referenced files exist:

```bash
python check_submission.py --draft
```

Run the full check before submitting. Use tags `<crsid>-q1` through `<crsid>-q4`
to identify your commits for each question. Replace `<crsid>` with your CRSid.
Push your branch and each tag to your own private GitLab project. For example,
after committing your final submission:

```bash
git tag bb667-q4
git push origin HEAD
git push origin bb667-q4
```

Push the `bb667-q1`, `bb667-q2` and `bb667-q3` tags in the same way. Here `origin`
must point to the assigned private GitLab project, not the public starter.
`git push origin HEAD` pushes the branch currently checked out. The tags
identify commits within your own repository; they do not give teammates access.

### Repository contents

```text
assignment.pdf              # coursework brief, included in the private handout
README.md                   # setup and submission instructions
requirements.txt            # CPU and submission-check dependencies
check_submission.py         # run before submitting
report/report.tex           # optional report skeleton
results/results.json        # numerical-results template, initially null
results.schema.json         # exact field names and types
src/strata/                 # supplied scientific code
```

The public starter omits `assignment.pdf`; use the brief distributed by the course.
The optional `.gitlab-ci.yml` runs the same check on the private GitLab project.
It checks `results/results.json` and reports incomplete submissions as failures.
Code, tests and plots must also be committed.
