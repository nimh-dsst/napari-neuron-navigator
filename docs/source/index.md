# Neuron Navigator

Neuron Navigator is a napari plugin for navigating, visualizing, and analyzing
neuronal morphology data from SWC and Parquet files.

This first documentation page provides a starting point for installing the
plugin. Detailed tutorials will be added here as the documentation grows.

## Get started

Install [Pixi](https://pixi.sh/latest/#installation) and
[Git](https://git-scm.com/downloads), then clone the repository:

```bash
git clone https://github.com/nimh-dsst/napari-neuron-navigator.git
cd napari-neuron-navigator
```

Install the plugin in the project's Pixi environment:

```bash
pixi run build
```

Launch napari with the plugin available:

```bash
pixi run napari
```

Pixi manages Python, napari, and the plugin's dependencies. The project supports
Windows, macOS on Apple Silicon, and Linux.

## Existing guides

- [Repository README](https://github.com/nimh-dsst/napari-neuron-navigator/blob/main/README.md):
  installation, capabilities, and development commands.
- [CPD2 workflow](https://github.com/nimh-dsst/napari-neuron-navigator/blob/main/docs/cpd2_workflow.md):
  a detailed data-processing walkthrough.
- [Use cases](https://github.com/nimh-dsst/napari-neuron-navigator/blob/main/USE_CASES.md):
  repeatable workflows and their manual verification status.

## Offline documentation

[Download this documentation as a PDF](https://nimh-dsst.github.io/napari-neuron-navigator/napari-neuron-navigator.pdf).
The PDF is generated from the same Markdown source as this page on every site
build. At this stage, it contains this introductory page; the existing guides
linked above remain separate documents.
