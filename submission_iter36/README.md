# Submission package — "Boundary and Interface Classes for Universal Chern Hamiltonians via Doubling and Jamming"

This directory is the clean journal-submission package assembled at
iter 36 of the doubling-and-jamming manuscript development cycle. It
contains the three submission PDFs (main paper, supplement, cover
letter), their LaTeX sources, every figure and table-input file the
two manuscripts reference, the two companion Python scripts, a
`requirements.txt`, and this README.

## 1. Project title

Boundary and Interface Classes for Universal Chern Hamiltonians via
Doubling and Jamming.

Corresponding author: Yitzchak Shmalo, Einstein Institute of
Mathematics, Hebrew University of Jerusalem.

## 2. File inventory

### 2.1. Manuscript files (six files)

  - `paper_main.tex` — main-paper LaTeX source (~2,834 lines).
  - `paper_main.pdf`  — compiled main paper (33 pages, ~1.08 MB).
  - `paper_supplement.tex` — supplement LaTeX source (~19,168
    lines).
  - `paper_supplement.pdf` — compiled supplement (~263 pages,
    ~4.84 MB; effective ~160 pages of load-bearing mathematics after
    skipping the audit/framing/navigation/inventory cluster flagged
    by the iter-33 skip-on-first-reading notices).
  - `cover_letter.tex` — stand-alone cover-letter LaTeX source
    (210 lines).
  - `cover_letter.pdf` — compiled cover letter (3 pages, ~135 KB).

### 2.2. Figure files (32 PNG files)

All figures referenced by `paper_main.tex` and `paper_supplement.tex`
via the `\safeincludegraphics{...}` macro:

```
fig_collar_support_histogram.png
fig_component_localization_ex23.png
fig_contour_margin_lower_bound.png
fig_copy_confinement_robustness.png
fig_ex1_boundary_localization.png
fig_ex1_bulk_spectrum.png
fig_ex1_chirality_flip.png
fig_ex1_edge_propagation.png
fig_ex1_edge_propagation_disorder.png
fig_ex1_eig_localization_vs_M.png
fig_ex1_jammed_spectrum.png
fig_ex1_leakage_vs_M.png
fig_ex1_mesh.png
fig_ex2_bulk_spectrum.png
fig_ex2_edge_propagation.png
fig_ex2_jammed_spectrum.png
fig_ex2_mesh.png
fig_ex3_bulk_spectrum.png
fig_ex3_edge_propagation.png
fig_ex3_jammed_spectrum.png
fig_ex3_mesh.png
fig_first_order_localization_loglog.png
fig_first_order_resolvent_loglog.png
fig_flux_spectral_flow_ex2.png
fig_leakage_loglog.png
fig_multiboundary_component_drifts.png
fig_pl_boundary_comparison.png
fig_pl_collar_scaling.png
fig_refinement_scaling.png
fig_resolvent_convergence.png
fig_resolvent_loglog.png
fig_window_separation_thresholds.png
```

Five of these figures (`fig_pl_boundary_comparison`,
`fig_resolvent_loglog`, `fig_first_order_resolvent_loglog`,
`fig_leakage_loglog`, `fig_first_order_localization_loglog`) are also
referenced from `paper_main.tex`; the remainder are referenced only
from `paper_supplement.tex`.

### 2.3. Table-input LaTeX files (9 files)

Files referenced from the two manuscripts via `\input{...}` (each
wrapped in an `\IfFileExists{...}{\input{...}}{...}` guard inside the
two `.tex` sources):

```
copy_confinement_robustness_table.tex
eig_localization_summary.tex
flux_spectral_flow_table.tex
mesh_stats_table.tex
numerics_summary.tex
periodic_triangular_certificate_table.tex
periodic_triangular_extension_certificate_table.tex
pl_boundary_comparison_table.tex
pl_collar_scaling_table.tex
```

### 2.4. Companion code (3 files)

  - `periodic_triangular_bulk_certificate.py` — closed-form bulk-gap
    and Chern-pairing certificate for the periodic triangular
    benchmark (the load-bearing implementation of
    `Cref{thm:periodic_triangular_chern}` / supplement
    `app:proof:thm:periodic_triangular_chern`); regenerates
    `periodic_triangular_certificate_table.tex` and
    `periodic_triangular_extension_certificate_table.tex` via the
    `--write-table` flag.
  - `run_companion_numerics_with_robustness.py` — companion
    numerical pipeline regenerating the supplement
    `\input` tables `mesh_stats_table.tex`,
    `numerics_summary.tex`, `pl_boundary_comparison_table.tex`,
    `pl_collar_scaling_table.tex`,
    `flux_spectral_flow_table.tex`,
    `eig_localization_summary.tex`,
    `copy_confinement_robustness_table.tex` and all
    `fig_*.png` figures from the three numerical examples.
  - `requirements.txt` — Python dependency pins.

## 3. Build instructions

The manuscript and supplement cross-reference each other via the
LaTeX `xr` package. The supplement reads `paper_main.aux` to resolve
`\Cref{M-...}` cross-document references, and the main paper reads
`paper_supplement.aux` to resolve `\Cref{S-...}` cross-document
references. The build therefore requires two passes per file in
alternating order. The following sequence is sufficient:

```bash
pdflatex -interaction=nonstopmode paper_main.tex
pdflatex -interaction=nonstopmode paper_supplement.tex
pdflatex -interaction=nonstopmode paper_main.tex
pdflatex -interaction=nonstopmode paper_supplement.tex
pdflatex -interaction=nonstopmode paper_main.tex
pdflatex -interaction=nonstopmode paper_supplement.tex
pdflatex -interaction=nonstopmode cover_letter.tex
pdflatex -interaction=nonstopmode cover_letter.tex
```

After this sequence the three PDFs should compile clean:

  - `paper_main.pdf`: 33 pages, zero LaTeX warnings, zero undefined
    references, zero undefined citations.
  - `paper_supplement.pdf`: 263 pages, zero LaTeX warnings, zero
    undefined references, zero undefined citations, zero
    multiply-defined warnings.
  - `cover_letter.pdf`: 3 pages, zero LaTeX warnings, zero overfull
    `\hbox`.

The pre-existing `OMS/cmtt/m/n` and `OMS/cmtt/m/it` font-shape
warnings on `paper_supplement.tex` and the documented `pdfTeX warning
(dest):` messages on `paper_main.tex` (from plain `xr` rather than
`xr-hyper`) are typographic / cross-document warnings only, not
content warnings, and do not affect correctness.

The companion numerics regeneration is optional and not required for
journal submission; the tables and figures are already present in
this package. To regenerate the periodic-triangular certificate
table:

```bash
python periodic_triangular_bulk_certificate.py --check --write-table
```

To regenerate the full companion-numerics pipeline (mesh data,
spectra, edge-propagation, robustness panels):

```bash
python run_companion_numerics_with_robustness.py
```

## 4. Cross-document references via `xr`

Both manuscripts load the LaTeX `xr` package
(`\usepackage{xr}`) and call `\externaldocument{...}` once with an
alias prefix:

  - `paper_main.tex` line 70: `\externaldocument[S-]{paper_supplement}`
    so every `\Cref{S-foo}` in the main paper resolves to
    `\label{foo}` in the supplement.
  - `paper_supplement.tex` line 85: `\externaldocument[M-]{paper_main}`
    so every `\Cref{M-foo}` in the supplement resolves to
    `\label{foo}` in the main paper.

The `S-` / `M-` aliases are stripped at lookup; supplement labels
themselves do NOT start with `S-`, and main labels themselves do NOT
start with `M-`. This is a deliberate xr convention so the
alias-prefix mechanism is unambiguous.

## 5. Online source repository

Both manuscripts cite the project source repository at:

  https://github.com/yspennstate/Doubling-and-Jamming-A-Universal-Chern-Hamiltonian-on-Triangulated-Surfaces-with-Boundary

The companion Python scripts in this package are the load-bearing
implementations of the closed-form periodic-triangular certificate
and the numerical-examples pipeline; the abstract and
`paper_supplement.tex` cite the same GitHub URL as the canonical
source.

## 6. The `[Journal name]` placeholder

The cover letter `cover_letter.tex` at line 36 contains the
addressee-line placeholder

```
\emph{[Journal name]}
```

This is an intentional parameter to be replaced with the target
journal name immediately before submission. The placeholder is
present in the iter-19 cover-letter draft and was deliberately
preserved through iters 19, 34, and 35; replacement is the only
external-to-audit pre-submission action recorded by the iter-35
final pre-submission audit (`app:final_audit_iter35`).

## 7. Cumulative result count

The manuscript and supplement together prove a substantial body of
internal results. The headline counts (as recorded by the iter-31
chapter summary, the iter-32 HP-citation inventory, the iter-33
condensation table, and the iter-35 final pre-submission audit) are:

  - **Internal HP-replacement theorems (iters 1-11; Part A
    periodic-benchmark closed-bulk):** 7+ proved unconditional
    results spanning closed-bulk gap and Chern integer for three
    geometries (flat-torus, hexagonal, square), the internal
    Bellissard-Connes Chern formula, and the internal Combes-Thomas
    exponential decay lemma.

  - **Delaunay-extension cluster (iters 23-31; Part A bounded-
    geometry attempt):** 18 rigorously proved results in a three-tier
    classification:
      - 14 TIER 1 unconditional results (quasi-uniform Delaunay
        first step, multi-flip Schur-test assembly, common-refinement
        existence, sub-complex refinement existence via weakened
        inclusion, and more).
      - 3 TIER 2 conditional theorems on a single shared
        sub-conjecture (`thm:delaunay_gap_stability_conjecture`,
        `thm:diff_dim_coherence_iter25`,
        `thm:three_step_composition_iter29`); these remain
        conditional after iter 35.
      - 1 TIER 3 proved negative result
        (`thm:block_obstruction_iter30`), establishing that a
        specific composition strategy cannot close the
        bounded-geometry gap.

  - **Open question:** 1 absolute open question
    (`rem:delaunay_open_question`) framing the residual
    bounded-geometry closed-bulk gap as the single research-hard
    external dependency on Higson-Prodan.

The iter-31 chapter summary (`app:delaunay_chapter_summary_iter31`)
records the cumulative status table and dependency-flow diagram for
the three-tier classification; the iter-32 HP-citation inventory
(`app:hp_inventory_iter32`) records the residual HP-citation
load-out; the iter-33 condensation table
(`tab:condensed_audit_summary_iter33`) summarizes the audit and
framing trail.

## 8. Package assembly note

This package was assembled at iter 36 as a clean snapshot of the
post-iter-35 submission state. It does NOT include the internal
development scaffolding:

  - the 555 `proof_loop_K` synthesis directories (internal
    scaffolding referenced by the iter-1-35 development trail);
  - LaTeX build artifacts (`.aux`, `.log`, `.out`, `.toc`,
    `_output.log`);
  - `iterN_summary.md` files, `DONE_*.txt` markers, helper audit
    scripts (`audit_*.py`, `check_loops.py`);
  - the iter-1 split-source files `main.tex` and
    `main_split_source.tex`.

The above scaffolding is preserved in the development repository at
the GitHub URL above for reproducibility, but is not part of the
journal submission package.

Documentation of the iter-36 packaging step (file inventory,
intentional exclusions, build verification) is recorded in
`paper_supplement.tex` under
`\subsection{Submission-package assembly (iter 36)}`
(label `app:submission_package_iter36`) inserted at the end of
`app:condensed_audit_summary_iter33`, immediately after the iter-35
audit subsection and immediately before the section divider
preceding `sec:schur_proofs`. A new theorem-map row in
`tab:main_theorem_map` of `paper_main.tex` points to this
subsection via the `xr` alias `S-app:submission_package_iter36`.
