# A sublattice dimensionality index — code and computed results

Code and data behind *A sublattice dimensionality index validated against deposited
metal halides* (Zhu). Everything reported in the paper can be regenerated from
what is here, given the structures, which are identified throughout by Crystallography
Open Database entry number rather than redistributed.

## What the index is

For a chosen metal–ligand sublattice, two metal centres are bridged when they share a
ligand within the first coordination shell. Each bridge carries a weight

    w = m · exp[−(d − d0) / λ]

with *m* the number of shared bridging ligands, *d* the metal–metal separation, *d0* the
shortest bridged separation in the structure and λ = 1.0 Å a coupling decay length. The
index is

    D = d_top + w_next / (w_next + w̃)

where `d_top` is the periodic rank of the bridging network — the rank of the lattice
translation subgroup under which the connected component maps onto itself — and the
fractional term measures how close the structure is to gaining the next dimension.

Two conventions matter and are applied throughout. The halides are taken as a single
bridging class, and the framework metals are taken as a single sublattice. Evaluating a
mixed-halide or mixed-metal structure one element at a time reports a fragment of its own
network; Section 3 of the paper quantifies the cost.

## Layout

    code/     the index and every script that produced a number in the paper
    data/     computed indices and screening results, as newline-delimited or plain JSON
    figures/  the rendered figures at 600 d.p.i., and the json the figure
              scripts read. The plotting scripts themselves are MATLAB and are
              not distributed; every number they draw is in data/ and code/.

### code

| file | what it does |
|---|---|
| `dimindex.py` | reference implementation of the index on a named sublattice |
| `cod_wide_index.py` | database-scale indexer; chooses the anion class per structure |
| `harvest_validation.py` | assembles the halide validation set from author-stated labels |
| `reindex_multihalide.py` | halides as one bridging class |
| `harvest_chalcogenides.py` | the chalcogenide transfer set |
| `reindex_chalc_multimetal.py` | framework metals as one sublattice |
| `baseline_and_scale.py` | stoichiometric baseline and the whole-network comparison |
| `benchmark_vs_ase.py` | comparison against rank determination and topology scaling |
| `benchmark_speed.py` | throughput and its dependence on metal count |
| `fix_wnext.py` | the two guards considered for the fractional term |
| `export_polyhedra.py` | coordination polyhedra and repeat directions for the Figure 1 panels |

### data

| file | contents |
|---|---|
| `validation_multihalide.json` | 703 halides with author-stated labels, indexed |
| `validation_allmetal.json` | the 444 framework halides under the all-metal convention |
| `validation_chalcogenides.json` | 158 chalcogenides, single-metal indexing |
| `validation_chalc_multimetal.json` | the same set with framework metals combined |
| `validation_guards.json` | the guard comparison for the fractional term |
| `baseline_scale.json` | stoichiometric baseline and whole-network ranks |
| `speed.json` | per-structure timings |
| `benchmark_vs_ase.json` | agreement with the reference algorithms |
| `jac_panels.json` | polyhedra, cell edges and lattice for the four Figure 1 structures |

## Reproducing the paper's numbers

Requires Python 3.11 with `pymatgen` (2026.5.18) and `numpy`; `ase` (3.26.0) only for the
whole-network comparison. Structures are fetched from the Crystallography Open Database by
entry number, or read from a local copy of the bulk archive.

    python code/harvest_validation.py <cache> validation_set.json
    python code/reindex_multihalide.py validation_set.json <cache> validation_multihalide.json
    python code/baseline_and_scale.py validation_multihalide.json <cache> baseline_scale.json
    python code/benchmark_speed.py <cache> speed.json

## Limits

Cells containing more than 90 metal centres are declined rather than evaluated; five of the
703 halides fall into that category. The index describes the sublattice it is given and
makes no claim about the rest of the structure — that restriction is the point of the
method, not a shortcoming of it.

## Licence

Code under the MIT licence; data under CC BY 4.0.
