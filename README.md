# Quantum Computing Portfolio

**Sennai Kahssai**  
*Quantum Computing · Molecular & Cellular Biology · Computational Drug Discovery*

---

## Overview

This repository contains my technical portfolio in **quantum computing for the life sciences**, focused on **computational drug discovery and molecular-scale modeling**.

Quantum technology is a broad field. My work is centered on three pillars that are directly relevant to biological and chemical systems:

- **Quantum Machine Learning (QML)**
- **Quantum Optimization**
- **Quantum Molecular Simulation**

Together, these form a unified framework for learning, simulating, and optimizing molecular systems — from binding energies to reaction pathways and biological function.

---

## Background and Focus

My training spans **Molecular & Cellular Biology, Physics, Chemistry, and Computation**.

I bring wet-lab experience in molecular biology, enzymology, genetics, and biochemistry together with classical and quantum computing, numerical modeling, and machine learning. I am currently completing a **Master’s degree in Quantum Computing**, alongside advanced training in **ML/AI**, with the objective of applying quantum-enabled methods to biological and chemical problems.

The guiding aim of this work is simple:

> **Use quantum and hybrid quantum–classical algorithms to model, simulate, and optimize molecular systems relevant to drug discovery and biological function.**

---

## Repository Structure

This repository is organized the way a computational molecular lab works.

### `projects/`
Executable Jupyter notebooks implementing:
- Quantum machine-learning models  
- Quantum optimization pipelines  
- Molecular Hamiltonians and variational solvers  
- Hybrid quantum–classical workflows  

Each notebook is self-documented via Markdown cells at the top of the file.
Project-specific dependencies and input requirements are documented alongside each project.

---

### `resources/theory/`
Mathematical and physical foundations:
- **`basics.md`** — qubits, states, operators, measurement
- **`algorithms.md`** — VQE, QAOA, QRC, and related methods
- **`models.md`** — reserved for Hamiltonian and encoding notes

---

## Scientific Orientation

The work here sits at the intersection of:

- Quantum computation  
- Molecular physics and chemistry  
- Machine learning  
- Biological systems  

The three pillars — **QML, Optimization, and Molecular Simulation** — are treated as components of a single computational stack for understanding and designing molecular systems in biology and medicine.

## Reproducibility and notebook checks

Create the shared environment for offline notebooks from the repository root:

```bash
conda env create -f environment.yml
conda activate quantum-projects
python .github/scripts/execute_notebooks.py
```

GitHub Actions also executes the offline notebooks on pushes and pull requests. The shared runner uses project-specific CI configurations; the chemistry workflows execute their bounded default scientific studies. Notebooks requiring Colab uploads, local forecasting spreadsheets, ADMET datasets, or Triple Alpha credentials are not run in CI. The three chemistry studies use their dedicated scientific workflows.

Coursework has been consolidated under `projects/`, including `QFT/`,
`Shor_Senay_Kahsay.ipynb`, and `Optimization/Unit_Commitment_MOPSO.ipynb`.
QFT and Shor retain their coursework implementations and are explicitly excluded
from automated execution pending their own refinement. A directory move does not
establish scientific or execution validation.

The [quantum chemistry studies](projects/Quanutm%20Chemistry%20/README.md) share a
pinned environment and retain separate scientific tests and notebook CI. Their saved
notebook outputs and benchmark evidence are available for review. The shared
runner delegates chemistry execution to those dedicated workflows and excludes generated
artifact notebooks from discovery.

For the ADMET project, create its environment from that project directory so its existing `requirements.txt` is found:

```bash
cd "projects/Modular_QEML_admet_Pipeline_Senay Kahsay"
conda env create -f environment.yml
conda activate admet-triplealpha
```

Set `TRIPLE_ALPHA_API_KEY` and `TRIPLE_ALPHA_RUN_URL` in your local environment before running the end-to-end notebook. Never store API credentials in a notebook or commit them.

## Connect With Me
- [LinkedIn](https://www.linkedin.com/in/senay-kahsay)
- [Medium](https://medium.com/@kahssai.sen)

> “We are the music while the music lasts.” — *T. S. Eliot*
