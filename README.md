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

GitHub Actions also executes the offline notebooks on pushes and pull requests. The shared runner uses project-specific CI configurations; the chemistry workflows execute their bounded default scientific studies. Legacy notebooks requiring uploads are excluded; the forecasting project has its own pinned offline scientific workflow. The ADMET scientific workflow independently acquires its pinned public dataset and runs offline tests, notebooks, and benchmarks. Live Triple Alpha calls are excluded from CI. The three chemistry studies use their dedicated scientific workflows.

The QFT folder contains two scientific studies: merged cross-framework Fourier
arithmetic and Shor order finding. Its dedicated workflow executes both notebooks,
independent phase/matrix references and reversible-arithmetic checks. See
[QFT arithmetic and Shor](projects/QFT/README.md) for reproducible execution and
compiled-oracle limits. Generated artifact notebooks are excluded from discovery.

The [grid-load QRC benchmarks](projects/Grid%20Load%20QRC%20Benchmarks/README.md)
compare CV and analogue-spin reservoirs with matched classical controls. Their
dedicated workflow verifies physics, leakage boundaries and both forecasting
notebooks on a licensed, checksummed NESO snapshot.

The [IROPS QUBO experiment](projects/QUBO%20IROPS%20solver/README.md)
combines exact capacity encoding, connecting-route expansion and decomposed
simulated annealing, with independently verified allocation, service tradeoffs
and logical resources. The client performs the industry comparison externally.

The [quantum chemistry studies](projects/Quanutm%20Chemistry%20/README.md) share a
pinned environment and retain separate scientific tests and notebook CI. Their saved
notebook outputs and benchmark evidence are available for review. The shared
runner delegates chemistry execution to those dedicated workflows and excludes generated
artifact notebooks from discovery.

For the ADMET project, create its environment from that project directory so its existing `requirements.txt` is found:

```bash
cd "projects/Quantum Molecular Learning for ADMET"
conda env create -f environment.yml
conda activate admet-triplealpha
```

The public ADMET notebook runs offline without credentials. Optional native execution requires verified deployment settings and local `TRIPLE_ALPHA_API_KEY` and `TRIPLE_ALPHA_RUN_URL` values. Never store API credentials in a notebook or commit them.

## Connect With Me
- [LinkedIn](https://www.linkedin.com/in/senay-kahsay)
- [Medium](https://medium.com/@kahssai.sen)

> “We are the music while the music lasts.” — *T. S. Eliot*
