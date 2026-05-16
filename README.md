# Doubling and Jamming Numerics

This repository contains the companion notebook and a standalone Python script for the numerical
experiments in the paper.

Run:

```bash
python run_companion_numerics_with_robustness.py
```

The script writes figures and LaTeX summary tables to `paper_outputs/`.  In addition to the
original mesh, doubled-gap, jammed-spectrum, localization, transport, finite-\(M\), collar, and
disorder checks, the current script includes three targeted diagnostics:

- comparison of the intrinsic Poincare-Lefschetz boundary Hamiltonian, the doubled original-side
  compression, and the finite-\(M\) jammed model;
- finite-size flux-insertion spectral flow for the annular example;
- subdivision scaling of the boundary-collar support of
  `H_partial - H_PL`.

The generated `paper_outputs/figures_and_tables.zip` bundles the rendered figures and tables.
