# Style Guide

This document defines the coding standards and conventions for the Thulium project.

## Code Formatting

### Black

All Python code must be formatted with Black.

```bash
black thulium tests
```

Configuration (`pyproject.toml`):

```toml
[tool.black]
line-length = 88
target-version = ['py310']
```

### Import Ordering

Imports are organized by isort (Black-compatible profile):

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from thulium.data.language_profiles import LanguageProfile
from thulium.pipeline.config import load_pipeline_config
```

---

## Type Hints

All public functions and methods must have type annotations.

```python
def recognize_image(
    path: Union[str, Path],
    language: str = "en",
    device: str = "auto"
) -> PageResult:
    """Recognize text in an image file."""
    ...
```

### Type Checking

Run mypy to verify type correctness:

```bash
mypy thulium
```

---

## Docstrings

Use Google-style docstrings for all public APIs.

```python
def get_language_profile(lang_code: str) -> LanguageProfile:
    """
    Retrieve the language profile for a given language code.

    This function provides access to the complete configuration for a
    supported language, including character set, tokenization strategy,
    and default decoder settings.

    Args:
        lang_code: ISO 639-1 language code (e.g., 'az', 'en', 'ru').

    Returns:
        LanguageProfile containing all configuration for the language.

    Raises:
        UnsupportedLanguageError: If the language code is not in the registry.

    Example:
        >>> profile = get_language_profile("az")
        >>> print(profile.name)
        Azerbaijani
    """
    ...
```

---

## Linting

### Ruff

Code is linted with Ruff for fast, comprehensive checks.

```bash
ruff check thulium tests
ruff check thulium tests --fix  # Auto-fix
```

Configuration (`pyproject.toml`):

```toml
[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E", "F", "B", "I"]
```

### Common Issues

| Code | Description | Fix |
| :--- | :--- | :--- |
| F401 | Unused import | Remove or add to `__all__` |
| F841 | Unused variable | Remove or prefix with `_` |
| E501 | Line too long | Reformat with Black |
| I001 | Import order | Run isort |

---

## Naming Conventions

### Modules and Packages

- Use lowercase with underscores: `language_profiles.py`

### Classes

- Use PascalCase: `HTRPipeline`, `LanguageProfile`

### Functions and Methods

- Use lowercase with underscores: `recognize_image()`, `get_language_profile()`

### Constants

- Use uppercase with underscores: `SUPPORTED_LANGUAGES`, `LATIN_BASE`

### Private Members

- Prefix with underscore: `_PIPELINE_CACHE`, `_build_profile()`

---

## Error Handling

### Custom Exceptions

Define domain-specific exceptions with clear messages:

```python
class UnsupportedLanguageError(ValueError):
    """Raised when a requested language is not supported."""

    def __init__(self, language_code: str, available_languages: List[str]):
        self.language_code = language_code
        self.available_languages = available_languages
        super().__init__(
            f"Language '{language_code}' is not supported. "
            f"Available: {', '.join(sorted(available_languages)[:10])}..."
        )
```

### Logging

Use structured logging via `structlog`:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Processing %s with language=%s", image_path, language)
logger.debug("Loaded image: %dx%d", image.width, image.height)
```

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
| :--- | :--- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Formatting, no logic change |
| `refactor` | Code restructuring |
| `test` | Adding or updating tests |
| `chore` | Build, CI, dependencies |

### Examples

```
feat(lang): add Georgian language profile

fix(pipeline): handle empty image input gracefully

docs(api): add examples to recognize_image docstring

refactor(models): extract common backbone interface
```

---

## Branching Strategy

- `main`: Stable, release-ready code
- `develop`: Integration branch for features
- `feature/<name>`: Individual feature branches
- `fix/<name>`: Bug fix branches

### Workflow

1. Create feature branch from `develop`
2. Implement and test changes
3. Open pull request to `develop`
4. Pass CI checks and code review
5. Merge to `develop`
6. Periodically merge `develop` to `main` for releases
