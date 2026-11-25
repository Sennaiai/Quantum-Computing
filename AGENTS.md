# Development and verification

- Discuss consequential design decisions and acceptance criteria with the human.
- For each task, add or update appropriate tests and run relevant checks.
  Report commands, results, failures, skipped checks, and remaining risks.
- The shared notebook check validates file structure only; it does not execute
  notebooks or verify scientific results. Add project-specific tests, controlled
  execution, dependency definitions, and reference comparisons where applicable.
- Protect private data and credentials. Use external AI review only on request
  and only with authorization for the information shared.
- Request human review before merging or releasing. Do not claim production
  readiness from a passing structural check.
- Keep scientific execution CI scoped to changes in each project. Do not rerun
  completed projects for unrelated work or shared notebook-routing changes;
  retain manual reruns and lightweight shared checks.
