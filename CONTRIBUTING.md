# Contributing to Thulium

Thank you for your interest in contributing to **Thulium**. We strive to build a state-of-the-art multilingual handwriting recognition library and welcome contributions from the research and engineering community.

This document provides guidelines for contributing to the project. Please read it carefully before submitting a Pull Request.

---

## 1. Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment for everyone.

---

## 2. Development Workflow

### 2.1 Environment Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/olaflaitinen/Thulium.git
   cd Thulium
   ```

2. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .[dev]
   ```
   This installs the package in editable mode along with all development dependencies (pytest, black, ruff, mypy).

### 2.2 Branching Strategy

- **main**: The stable production branch.
- **develop**: The integration branch for next release (if applicable).
- **feature/***`: Feature branches derived from main.
- **fix/***`: Bug fix branches.

### 2.3 Style Guide

We enforce strict quality standards to ensure maintainability and robustness.

- **Formatter**: [Black](https://github.com/psf/black) (line length: 88)
- **Linter**: [Ruff](https://github.com/astral-sh/ruff)
- **Type Checking**: [Mypy](http://mypy-lang.org/) (strict mode preferred)

Run the quality suite before committing:

```bash
# Format code
black thulium tests

# Lint code
ruff check thulium tests

# Type check
mypy thulium
```

---

## 3. Testing Standards

All contributions must include comprehensive tests.

### 3.1 Test Framework

We use `pytest` for all testing. Tests are located in the `tests/` directory.

### 3.2 Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_language_profiles.py

# Run with coverage
pytest --cov=thulium
```

### 3.3 Coverage Requirements

- New features must have 100% test coverage.
- Bug fixes must include a regression test.

---

## 4. Pull Request Process

1. **Title**: Use [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat: add attention decoder`, `fix: sequence length error`).
2. **Description**: clear explanation of changes, reasoning, and any breaking changes.
3. **Verification**: Confirm that all tests pass and linting is clean.
4. **Review**: Wait for maintainers to review. Address feedback promptly.

---

## 5. Adding New Languages

To add support for a new language:

1. **Research**: Identify the script, alphabet, and writing direction.
2. **Profile**: Add a new `LanguageProfile` in `thulium/data/language_profiles.py`.
3. **Model**: Assign an appropriate `model_profile` (shared or new).
4. **Test**: Add validation cases in `tests/test_language_profiles.py`.
5. **Docs**: Update `docs/models/language_support.md`.

---

## 6. License

By contributing to Thulium, you agree that your contributions will be licensed under the Apache License 2.0.
