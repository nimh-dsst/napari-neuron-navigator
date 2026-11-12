# Documentation

The published site is at
[nimh-dsst.github.io/napari-neuron-navigator](https://nimh-dsst.github.io/napari-neuron-navigator/).
Its source lives in `docs/source/`. The initial site contains one introductory
Markdown page, `docs/source/index.md`.

## Build and preview

Run these commands from the repository root:

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve
```

The build generates the PDF first, then builds HTML with Sphinx warnings treated
as errors. The preview serves `docs/_build/html/` at <http://127.0.0.1:8000>;
press Ctrl+C to stop it. Re-run after editing the sources.

The `docs` Pixi feature is a separate dependency group and environment with
`no-default-feature = true`: building documentation does not install napari or
Qt. It follows Chronoxiv's stack: Sphinx 8.1, `pydata-sphinx-theme`,
`sphinx-copybutton`, and `sphinx-design`. MyST-Parser adds Markdown support.
WeasyPrint, Markdown-It, and Pygments provide the existing PDF conversion.
Pixi installs this environment on first use; `pixi.lock` records its dependencies.

## Markdown to PDF

Generate only the published documentation PDF:

```bash
pixi run -e docs docs-pdf
```

This converts `docs/source/index.md` into
`docs/_build/pdf/napari-neuron-navigator.pdf`. `docs-build` also copies it to the
HTML site's root for downloading. The PDF currently covers only the introductory
page. Extend the PDF build when adding more published pages.

Convert any standalone Markdown document using the same workflow:

```bash
pixi run -e docs markdown-pdf path/to/input.md docs/_build/output.pdf
```

Use ordinary Markdown headings, links, lists, and fenced code blocks in content
shared with the PDF converter. Sphinx/MyST directives and roles are available
for HTML pages but are not interpreted by the standalone converter.

All generated HTML and PDFs stay under the gitignored `docs/_build/` directory.

## Publishing

`.github/workflows/docs.yml` builds HTML and PDF on documentation pull requests,
relevant pushes to `docs/tutorial` and `main`, and manual runs. Every build uploads
the PDF as a `documentation-pdf` Actions artifact. Pushes and manual runs on those
two branches deploy the complete HTML site, including the downloadable PDF,
using GitHub's official Pages artifact and deployment actions.

GitHub Pages must use **GitHub Actions** as its publishing source. The
`github-pages` environment allows `docs/tutorial` for this initial publication
and `main` for publication after merge. When the tutorial branch is retired,
remove it from the workflow's push/deploy conditions and the environment's
deployment branch rules. Pull requests build and upload artifacts without
deploying.

See [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
for the deployment permissions and environment requirements.

## Data Caution: SWC `type` Does Not Reliably Identify The Compartment

**Some neurons in `isocortex_total_right_brainglobe_flatmap.parquet` have
dendritic projections typed `2` (axon).** This was found on 2026-08-05 by
visually inspecting detected termini in napari.

Anything in these documents that filters or groups by node `type` — axon
termini, axon-only projections, compartment-specific heatmaps — inherits this
problem. The plugin reports exactly what the source files say; the source files
are wrong for an unmeasured subset of neurons.

When reading or writing docs here:

- Say **axon-typed**, not **axon**. A count of axon termini is an upper bound
  contaminated by an unknown number of dendrite tips.
- Do not claim node types cleanly separate axonal from dendritic arbors.
- Verify compartment-based results visually before treating them as biology.
- The extent is unquantified — do not state a rate until someone measures one.

Purely topological results (childless tests, branch points, path lengths) are
unaffected as long as they do not filter or interpret by `type`. See `AGENTS.md`
for the full rule and `USE_CASES.md` UC-010 for the workflow where this surfaced.
