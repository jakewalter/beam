Contributing
============

Thanks for thinking about contributing! Here are the basics to get started:

- Run the test suite:
  ```bash
  PYTHONPATH=. python3 -m pytest -q
  ```
- Run the pipeline for a sample day (example)
  ```bash
  PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/template_config.json --dry-run
  ```
- Formatting
  - Use `black` + `isort` for formatting the code.
- Create issues or PRs with a clear description and tests if adding behavior.

Repository maintainers may ask you to add tests or documentation for major changes.
