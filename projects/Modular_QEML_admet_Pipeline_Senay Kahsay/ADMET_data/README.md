# ADMET Future Curation

Cleaned notebook for ADMET dataset curation and Hamiltonian construction only.

Responsibilities:
- Load ADMET datasets
- Generate Morgan fingerprints
- Select features with mutual information
- Build feature-interaction matrices
- Construct Hamiltonian A, B, and C dictionaries
- Export curated Hamiltonian payloads for the class-based LCU framework

Excluded by design:
- LCU generation
- QBE generation
- Block encoding
- ReflectZero / WalkOperator
- Triple Alpha tests
