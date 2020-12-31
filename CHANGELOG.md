# evOWLuator Changelog

All notable changes to evOWLuator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
evOWLuator adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2020-12-30
### Added
- Energy evaluation via `powertop`.
- Docker-based install.

### Changed
- Improved labelling of csv correctness reports.
- Moved `evowluate` script to `bin` dir.

### Fixed
- Scatterplots now correctly show all available datapoints for each reasoner.

## [0.1.1] - 2020-03-02
### Added
- Legend configuration arguments (`--legend-cols`, `--legend-only`).
- Color configuration arguments (`--colors`).
- Marker configuration arguments (`--markers`, `--marker-size`).
- Label rotation arguments (`--label-rot`, `--xtick-rot`, `--ytick-rot`).

### Changed
- Reasoners are plotted respecting the order given via the `-r` argument.
- Value labels have some padding with respect to histogram bars.

### Removed
- Broken label fitting logic.

### Fixed
- Reasoners omitted via the `-r` argument are not shown in summaries.

## [0.1.0] - 2019-10-16
### Added
- Support for classification, consistency and matchmaking inference tasks.
- Correctness, performance and energy evaluation modules.
- Dataset syntax conversion functionality.
- Evaluation results visualization.
- HTML documentation.
- Host platforms: Linux, macOS, Windows (via WSL).

[0.1.2]: https://github.com/sisinflab-swot/evowluator/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sisinflab-swot/evowluator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sisinflab-swot/evowluator/releases/tag/v0.1.0
