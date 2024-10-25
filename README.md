<!--
SPDX-FileCopyrightText: 2024 2024 Christopher C. Holst, KIT

SPDX-License-Identifier: GPL-3.0-only
-->

![License checks](https://github.com/ccholst/PALM-static_tool/actions/workflows/license_checks.yml/badge.svg)
![Unit tests](https://github.com/ccholst/PALM-static_tool/actions/workflows/unittests.yml/badge.svg)
![Linting](https://github.com/ccholst/PALM-static_tool/actions/workflows/linting.yml/badge.svg)

# PALM-static_tool
This helps modify PALM-4U <JOB_ID>_static files for use with offline nesting (mesoscale forcing lateral bonudary conditions) in realistic domains.

The code adds several optional modifications:

- change `pavement_type` 7 to 1
- change `pavement_type` > 15 to 1
- smooth boundary topography along the boundary
- ramp building heights near the boundary
- create plots to verify results

## Pavement types

In the past, there was a bug with metal surfaces causing large heat fluxes. While this has been fixed, for most domains and grid spacings, metal surfaces are seldom found, so this function is kept in the tool for reference.

Depending on the input data, pavement type values might sometimes be out of range. There is a catch-all for that too.

## Topography heights

The offline-nested "parent" domain of the PALM-4U domain is typically smooth on scales of 1000s of meters. To account for this and potential mass flux and profile gradient problems, the PALM-4U domain boundaries are smoothed tangentially, using moving smoothing functions.

By default, the aggregation is done by the `numpy.median` function.

The procedure is as follows:

- Extract all boundary values from the 2-D topography array
- Vectorize clockwise
- Buffer at the beginning and end of the vector (i.e., cyclic extension)
- Apply moving aggregation to a copy of the vector (think: Nokia Snake cell phone game)
- "Unwind" the vector back onto the 2-D grid
- Repeat for the next "frame" of grid points, which is neighboring to the inside of the domain

Simple Visualization:

```
        ->   top
      o----------o
      |  ->      |
    l |  o----o  | r
    e |  |    |  | i
    f |  |    |  | g
    t |  |    |  | h
      |  o----o  | t
      |          |
      o----------o
         bottom
```


This process virtually creates a triangular smoothing kernel, i.e., smooth on longer tangential scales near the boundary and shorter tangential scales further inside the domain. This works for realistic (smooth-ish) topography. For rugged cliffs, arguably, the boundaries should be extended to avoid those.

## Building heights

A linear ramp is applied from 0 at the first grid point to the correct building height at a distance `BUFFER_B` from the boundary.

Currently, only `buildings_2d` can be modified correctly. There is a function for `buildings_3d` in the code, but PALM raises errors when this is employed.

> [!TIP]
> It is also not necessarily useful to add `building_3d` functionality: mesoscale offline nesting is typically applied to grid spacings, where 3-D building structures do not make much sense to simulate in most domains (i.e., > 20 meters). If one were to simulate the largest structures on earth, those would be far from the domain boundary and be unaffected by modifications.

## Plots

The results are then plotted and saved. Two plots are created by default:

- topography height
- topography height adjustment

Both include additional landuse information like water bodies and buildings for orientation.

Example plots:

[Topography height](/docs/assets/_static_map_TEST.png "Map")

[Topography adjustment](/docs/assets/_static_bdy_TEST.png "Adjustment")

---
# Testing and code contributions

To contribute, please copy the `z_t_bdy{X}_debug` function to create `z_t_bdy{X+1}_debug`, before adding the `z_t_bdy{X+1}` function and modifying the calls. This helps maintaining functionality, while experimenting.

Unit tests are implemented for the topography part of the code. It is currently done by having a copy of the function with excessive diagnostic output and assertions, which is called with two seperate test flags, one for the outermost "ring" and one for inner rings. These tests will fail, if the code produces uncorrect results, even though the implementation is unconventional.

If in the future more complex behavior is added in other segments of the code, additional testing will be implemented. For the current level of complexity, this seems unnecessary.

---

# Licenses

Copyright &copy; 2024 Christopher Claus Holst, KIT

License for code: GPL-3.0-only

License for documentation and assets: CC-BY-4.0
