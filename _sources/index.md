:::{image} _static/logo.svg
:class: only-light
:alt: SciTiff
:width: 60%
:align: center
:::
:::{image} _static/logo-dark.svg
:class: only-dark
:alt: SciTiff
:width: 60%
:align: center
:::

```{raw} html
   <style>
    .transparent {display: none; visibility: hidden;}
    .transparent + a.headerlink {display: none; visibility: hidden;}
   </style>
```

```{role} transparent
```

# {transparent}`SciTiff`

<div style="font-size:1.2em;font-style:italic;color:var(--pst-color-text-muted);text-align:center;">
  Scientific tiff format for imaging experiments.
  </br></br>
</div>

:::{include} user-guide/installation.md
:heading-offset: 1
:::

## Get in touch

- If you have questions that are not answered by these documentation pages, ask on [discussions](https://github.com/scipp/scitiff/discussions). Please include a self-contained reproducible example if possible.
- Report bugs (including unclear, missing, or wrong documentation!), suggest features or view the source code [on GitHub](https://github.com/scipp/scitiff).

SciTiff format inherits [``HyperStacks``](https://imagejdocu.list.lu/gui/image/hyperstacks) and define metadata on top of the ``HyperStacks``.

`scitiff` project hosts both metadata schema and `io` helpers to save/load scitiff images.

| Supported IO Languages | Metadata Schema Format |
| ---------------------- | ---------------------- |
| python                 | json                   |

> Currently there is only `python` io modules.

## Why SciTiff?
`HyperStacks` has been a standard format for high energy imaging experiments.
It defines dimensions of image stack (`t`, `z`, `c`, `y`, `x`).
However, it does not guarantee the order of dimensions.
Also, there is no standard way of storing the coordinate of each dimension.

For example, if there are 1_000 tiff images along the `t` dimension, it is not clear if it is time of flight, or wall clock time or if it is every 1 ns or every 1 Å or if the interval is non-uniform or etc...

Therefore `scitiff` project aims to define a consistent way of storing the physical properties of a tiff image stack as metadata.

## Quick Links

:::{card} IO User Guide
:link: user-guide/io.html
:::

## Scitiff Metadata Schema
Metadata is stored as a plain text `json` so the schema is defined as a `json schema`.

Here is an example of the scitiff metadata of a tiff file.

:::{literalinclude} ./_static/example_metadata.json
  :language: JSON
:::

```{warning}
  Currently it is not allowed to have multi-dimensional coordinates in the metadata.<br>
  All coordinate should be a `single` or `zero` dimensional data.<br>
  For example, if you want to store `event_id`, which is folded into `(x, y)`, it is not possible.<br>
  It is because we don't want to store huge coordinate values as a plain text,<br>
  which can make a tiff file size unnecessarily large and make image loading slow.

```

SciTiff Metadata Schema is written based on the [`scipp.DataArray`](https://scipp.github.io/user-guide/data-structures/data-structures.html#DataArray) data structure.
You can (almost) directly turn the `image` field of the metadata into a `scipp.DataArray`.<br>
But the `values` of the `image` is supposedly stored as tiff stack.

```{note}
  The metadata schema is defined as a [`pydantic.Model`](https://docs.pydantic.dev/latest/concepts/models/) and exported as a plain text json so that any platform can use the schema.

  See source code of the [`ScitiffMetadataContainer`](https://ess-dmsc-dram.github.io/scitiff/_modules/scitiff/_schema.html#SciTiffMetadataContainer) to see the pydantic model definition.
```

## Download Scitiff Metadata Schema
{download}`Scitiff Metadata Schema Json File <./_static/metadata-schema.json>`

Here is the full metadata schema as a plain text.

:::{literalinclude} ./_static/metadata-schema.json
  :language: JSON
:::

```{toctree}
---
hidden:
---

user-guide/index
api-reference/index
developer/index
about/index
user-guide/index
```
