# napari-neuron-navigator

A napari plugin for navigating, visualizing, and analyzing neuronal morphology
data from SWC and Parquet files.

[![Tests](https://github.com/nimh-dsst/napari-neuron-navigator/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/nimh-dsst/napari-neuron-navigator/actions/workflows/test.yml)
[![Documentation](https://github.com/nimh-dsst/napari-neuron-navigator/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/nimh-dsst/napari-neuron-navigator/actions/workflows/docs.yml)

[Documentation](https://nimh-dsst.github.io/napari-neuron-navigator/) ·
[PDF](https://nimh-dsst.github.io/napari-neuron-navigator/napari-neuron-navigator.pdf)

## Isocortex dataset

The isocortex dataset is available on Zenodo:
[DOI: 10.5281/zenodo.23960862](https://doi.org/10.5281/zenodo.23960862).

## Prerequisites

Install [Pixi](https://pixi.sh/latest/#installation) and
[Git](https://git-scm.com/downloads). The project supports Windows, macOS on
Apple Silicon, and Linux.

Pixi manages Python 3.11 or 3.12, napari 0.9.x, and the project's dependencies
using the repository lock file.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/nimh-dsst/napari-neuron-navigator.git
cd napari-neuron-navigator
```

Build and install the package in editable mode:

```bash
pixi run build
```

Editable mode allows you to make changes without reinstalling.

## Running

Launch napari with the plugin installed:

```bash
pixi run napari
```

This command automatically builds the package if needed before launching napari.

See the [documentation](https://nimh-dsst.github.io/napari-neuron-navigator/)
for workflows, debug logging, testing, compatibility, and attributions.
