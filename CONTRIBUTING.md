# Contributing to MASCAF

Thank you for your interest in contributing! All contributions — bug reports, feature requests, documentation improvements, and code — are welcome.

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Reporting bugs

Open an issue on [GitHub Issues](https://github.com/jmrfox/mascaf/issues). Include:

- A short description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## Suggesting features

Open an issue and describe the use case you have in mind. Explaining *why* the feature would be useful helps maintainers prioritize it.

## Submitting a pull request

1. **Fork** the repository and create a branch from `main`.
2. Make your changes, following the existing code style.
3. Add or update tests in `tests/` if relevant.
4. Run the test suite locally to confirm everything passes:

   ```bash
   uv sync
   uv run pytest
   ```

5. Open a pull request against `main` with a clear description of what you changed and why.

## Development setup

```bash
git clone https://github.com/jmrfox/mascaf.git
cd mascaf
uv sync
uv run pytest
```

The optional CGAL-backed preprocessing requires a native C++ toolchain — see the README for details. It is not needed for most contributions.
