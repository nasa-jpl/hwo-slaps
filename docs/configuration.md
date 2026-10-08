# Configuration reference

This page lists every configuration key, generated from the tables HWO-SLAPS uses to
validate configurations. Each table gives the key, the allowed values, the default, the
unit and the meaning. Bulleted lines under a table are rules that involve several keys.

[Configuration files](guide/configuration.md) explains the format and how files are
combined. You can print any section from the command line:

```bash
hwoslaps reference                   # everything
hwoslaps reference psf.truth         # one section
```

Headings name the path of each section. `<name>` stands for a component name you choose,
`[i]` for a list item, and `(type: ...)` or `(kind: ...)` for the alternative a table
describes.

The sections below `top level` describe a configuration file. Later sections describe
the other inputs: `population` and `batch` for [batch files](guide/batches.md), and `fit`,
`sampler`, `refine` and `classification` for [nonlinear fits](guide/nonlinear.md). The
`config` key of a batch file takes a complete configuration with the same keys as the
`top level` section.

<!-- GENERATED_CONFIGURATION -->
