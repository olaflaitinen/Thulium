# API Reference

This section documents the public API of Thulium.

## Core API (`thulium.api`)

The primary entry points for interacting with the library.

### Recognition

- [`recognize_image`](api_high_level.md#recognize_image): Process a single image path or array.
- [`recognize_batch`](api_high_level.md#recognize_batch): Process a list of images in parallel.
- [`recognize_pdf`](api_high_level.md#recognize_pdf): Extract and recognize text from PDF pages.

### Types

- [`PageResult`](api_high_level.md#pageresult): Container for page-level recognition results.
- [`LineResult`](api_high_level.md#lineresult): Container for line-level text and confidence.

---

## Data Layer (`thulium.data`)

### Language Profiles

- [`get_language_profile`](../../thulium/data/language_profiles.py): Retrieve configuration for a specific language.
- [`list_supported_languages`](../../thulium/data/language_profiles.py): Get list of all supported ISO codes.
- [`LanguageProfile`](../../thulium/data/language_profiles.py): Dataclass defining alphabet, script, and model config.

---

## Pipeline (`thulium.pipeline`)

- [`HTRPipeline`](../../thulium/pipeline/htr_pipeline.py): Configurable processing pipeline (Preproc -> Seg -> Recog -> Decode).

---

## Evaluation (`thulium.evaluation`)

- [`cer`](../../thulium/evaluation/metrics.py): Character Error Rate computation.
- [`wer`](../../thulium/evaluation/metrics.py): Word Error Rate computation.
- [`ser`](../../thulium/evaluation/metrics.py): Sequence Error Rate computation.

For comprehensive architectural details, see [Architecture](../architecture.md).
