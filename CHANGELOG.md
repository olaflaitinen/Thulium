# Changelog

All notable changes to Thulium will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-11

### Added

**Language Parity**
- Complete language profiles for 52+ languages across 12 writing systems
- First-class support for every language with explicit configuration and examples
- Regional language groups: Scandinavian, Baltic, Caucasus, Western Europe, Eastern Europe, Middle East, South Asia, East Asia, Southeast Asia, Africa
- Model profile assignments for fair language-to-model mapping

**Multilingual Model Configurations**
- `htr_latin_multilingual.yaml`: Shared Latin model for 35+ languages
- `htr_cyrillic_multilingual.yaml`: Russian, Ukrainian, Bulgarian, Serbian
- `htr_arabic_multilingual.yaml`: Arabic, Persian, Urdu (RTL)
- `htr_georgian.yaml`: Specialized Georgian script model
- `htr_armenian.yaml`: Specialized Armenian script model
- `htr_cjk_multilingual.yaml`: Chinese, Japanese, Korean
- `htr_indic_multilingual.yaml`: Hindi, Bengali, Tamil, Telugu, etc.

**Advanced Decoders**
- Transformer-based attention decoder with beam search inference
- CTC decoder with language model integration and beam search
- Configurable BeamSearchConfig with alpha/beta coefficients

**Model Architectures**
- ResNet variants (resnet18, resnet34, resnet_small, resnet_tiny)
- Lightweight CNN backbone with depthwise separable convolutions
- Vision Transformer (ViT) backbone with patch embedding
- Hybrid CNN-ViT backbone
- BiLSTM, stacked LSTM, and LSTM with attention heads
- Transformer and Conformer sequence heads

**Language Models**
- Neural character language model (LSTM and Transformer variants)
- Character-level n-gram LM with smoothing (add-k, interpolation, backoff)
- Word-level n-gram LM
- Incremental scoring for efficient beam search

**Benchmarking Suite**
- `BenchmarkConfig` and `BenchmarkResult` dataclasses
- Per-language metric aggregation
- Latency and throughput measurement
- Report generation (Markdown, CSV, JSON, HTML)
- Comparison utilities across models

**Error Analysis**
- Character confusion matrix generation
- Edit operation classification (substitution, deletion, insertion)
- Per-language error aggregation
- Common error pattern identification

**Noise Injection**
- Gaussian and salt-and-pepper noise
- Blur and JPEG compression artifacts
- Random occlusions
- Resolution degradation

**Training Configurations**
- `htr_latin_multilingual.yaml` training config
- Fine-tuning template for per-language optimization
- Low-resource transfer learning config

**Examples**
- `recognize_scandinavian.py`: Nordic languages demo
- `recognize_caucasus.py`: Caucasus region demo
- `recognize_baltic.py`: Baltic languages demo
- `recognize_multilingual.py`: Full 52+ language demo

**Documentation**
- Comprehensive `docs/evaluation/benchmarks.md` with per-language results
- Fairness metrics (CER variance, standard deviation)
- Theoretical foundations in `docs/theory/htr_foundations.md`

### Changed
- Updated to Beta status (Development Status :: 4 - Beta)
- Version bumped to 0.2.0
- Expanded PyPI classifiers for better discoverability
- Added `editdistance` to core dependencies

### Fixed
- Language profile validation for all 52 languages
- Model profile field added to LanguageProfile dataclass

---

## [0.1.0] - 2024-11-15

### Added
- Initial release of Thulium
- Core HTR pipeline with preprocessing, segmentation, and recognition
- Basic CNN backbone and LSTM sequence modeling
- CTC decoder with greedy decoding
- Support for major Latin-script languages
- CLI interface with recognize and benchmark commands
- Configuration system with YAML files
- Basic documentation and README

### Notes
- Alpha release - APIs subject to change
- Focus on English and Western European languages
