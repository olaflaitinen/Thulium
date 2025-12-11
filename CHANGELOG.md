# Changelog

All notable changes to Thulium will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2024-12-11

**Production-Ready Release**

This release marks Thulium as production-ready with comprehensive multilingual support, state-of-the-art architectures, and professional documentation.

### Added

**Documentation Enhancement**
- Complete README rewrite with professional badges and Mermaid architecture diagrams
- Enhanced `docs/architecture.md` with class diagrams, component hierarchies, and mathematical formulas
- Comprehensive `docs/evaluation/metrics.md` with fairness metrics and statistical analysis
- Detailed benchmark documentation with per-language performance tables
- Academic citation block for research publications

**Architecture Diagrams**
- System-level architecture diagram with all processing layers
- Model architecture diagram showing backbone, sequence head, and decoder variants
- Pipeline flow diagrams with preprocessing, segmentation, and recognition stages
- Evaluation framework diagram with metrics and analysis components
- XAI layer explainability diagram

**Mathematical Formulations**
- CER/WER/SER metric definitions with formal notation
- CTC loss formulation with path marginalization
- Attention mechanism equations (scaled dot-product, multi-head)
- Language model scoring formulas for beam search
- Cross-language fairness metrics (Delta_CER, Sigma_CER)

### Changed

- Version upgraded to 1.0.0 (Production/Stable)
- Development Status classifier updated from Beta to Production/Stable
- PyPI package name: `thulium-htr`
- README restructured with collapsible language sections
- All badges updated with flat-square style and correct package name

### Language Support

Complete first-class support for 56 languages across 12 writing systems:

| Region | Languages |
|:-------|:----------|
| Scandinavian | Norwegian (Bokmal, Nynorsk), Swedish, Danish, Icelandic, Faroese, Finnish |
| Baltic | Lithuanian, Latvian, Estonian |
| Caucasus | Azerbaijani, Turkish, Georgian, Armenian |
| Western Europe | English, German, French, Spanish, Portuguese, Italian, Dutch |
| Eastern Europe | Polish, Czech, Slovak, Hungarian, Romanian, Croatian, Slovenian, Russian, Ukrainian, Bulgarian, Serbian, Greek |
| Middle East | Arabic, Persian, Urdu, Hebrew |
| South Asia | Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Punjabi, Kannada, Malayalam |
| East Asia | Chinese, Japanese, Korean |
| Southeast Asia | Thai, Vietnamese, Indonesian, Malay |
| Africa | Swahili, Afrikaans |

---

## [0.2.0] - 2024-12-11

### Added

- Language profiles for 52+ languages
- Model profile field in LanguageProfile dataclass
- Multilingual model configurations (Latin, Cyrillic, Arabic, Georgian, Armenian, CJK, Indic)
- Training configurations for multilingual and fine-tuning workflows
- Per-region example scripts
- Comprehensive test suite for language profiles
- Benchmark documentation with fairness metrics

### Changed

- Version upgraded to 0.2.0 (Beta)
- Added `editdistance` to core dependencies

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
