# Local ADMET Inputs

This directory contains the ten tab-separated ADMET datasets used by the
canonical curation notebook at `../pipeline/ADMET_future_curation_complete.ipynb`.
The notebook loads these files locally; it does not download them at runtime.

The raw inputs are in `data/`. The curation notebook produces the intermediate
fingerprints, selected features, interaction data, and Hamiltonian handoff used
by the counterdiabatic and Triple Alpha workflows.
