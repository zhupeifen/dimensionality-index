# A sublattice dimensionality index — code and computed results

Code and data behind *A sublattice dimensionality index validated against deposited
halides and chalcogenides* (Zhu). Everything reported in the paper can be regenerated
from what is here, given the structures, which are identified throughout by
Crystallography Open Database entry number rather than redistributed.

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

Three numerical parameters, not two. Besides `r_c` and λ, non-bridged candidate pairs are
searched to a radius `R_next = 8.0 Å`; the candidate carries unit multiplicity, where an
internal bridge carries *m*. `code/sweep_rnext.py` measures what `R_next` changes: nothing
in the integer rank at any radius from 4 to 12 Å, and a great deal in the fractional term.

The fractional term is defined only where a bridging network exists. A structure with no
metal–ligand–metal bridge has no median coupling to compare against and is reported with
rank zero and no fractional part, not with a fractional part of zero; 101 of the 703
halides are in that category and are held out of any zero-versus-nonzero comparison.

Two conventions matter and are applied throughout. The halides are taken as a single
bridging class, and the framework metals are taken as a single sublattice, with the alkali
metals, the alkaline earths and thallium excluded as A-site spectators. Evaluating a
mixed-halide or mixed-metal structure one element at a time reports a fragment of its own
network; Section 3 of the paper quantifies the cost.

## Layout

    code/     the index and every script that produced a number in the paper
    data/     computed indices and screening results, as plain JSON
    figures/  the rendered figures at 750 d.p.i., and the JSON the figure scripts read
    MANIFEST.json   every file above with its size and SHA-256

The plotting scripts are MATLAB and belong to an internal library that is not
distributed. Every value they draw is in `figures/*.json` and `data/`, so each figure can
be redrawn in any tool from what is here.

### code

| file | what it does |
|---|---|
| `dimindex.py` | reference implementation of the index on a named sublattice |
| `cod_wide_index.py` | database-scale indexer; chooses the anion class per structure |
| `harvest_validation.py` | assembles the halide validation set from author-stated labels |
| `harvest_chalcogenides.py` | the chalcogenide transfer set |
| `reindex_multihalide.py` | halides as one bridging class |
| `reindex_allmetal.py` | **both conventions on all 703 halides**; produces the paper's validation numbers |
| `reindex_chalc_multimetal.py` | the same two conventions on the chalcogenides |
| `codscreen_wide.py` | the inorganic A(x)Cu(y)X(z) screen, binaries split by oxidation state |
| `codscreen_hybrid.py` | the wider screen admitting organic A-site cations |
| `baseline_and_scale.py` | stoichiometric baseline and the whole-network comparison |
| `wholenet_scale.py` | whole-network ranks at database scale |
| `benchmark_vs_ase.py` | comparison against rank determination and topology scaling |
| `benchmark_speed.py` | throughput and its dependence on metal count |
| `fix_wnext.py` | the two guards considered for the fractional term |
| `sweep_rnext.py` | sensitivity of rank and fractional term to the search radius `R_next` |
| `frac_stratified.py` | the fractional term against author labels, stratified by computed rank |
| `export_polyhedra.py` | coordination polyhedra and repeat directions for the Figure 1 panels |
| `retile_panel.py` | re-tiles a panel's primitive cell for the Figure 1 views |
| `fetch_cifs.py` | completes a local CIF cache from COD by entry number; run it first |

### data

| file | contents |
|---|---|
| `validation_allmetal703.json` | **the paper's validation set**: 703 halides under both conventions |
| `validation_multihalide.json` | the same 703 with halides combined but metals taken singly |
| `validation_allmetal.json` | the 444 framework halides, an earlier partial application |
| `validation_guards.json` | the guard comparison for the fractional term |
| `validation_chalcogenides.json` | 158 chalcogenides, single-metal indexing |
| `validation_chalc_multimetal.json` | the same set with framework metals combined |
| `sweep_rnext.json` | every structure re-indexed at seven values of `R_next` |
| `cod_hybrid.json` | the 3,512-structure hybrid screen, one record per structure |
| `screen_summary_wide.json` | counts and index values for the inorganic screen, as `codscreen_wide.py` writes them |
| `baseline_scale.json` | stoichiometric baseline and whole-network ranks |
| `wholenet_scale.json` | whole-network ranks across the database-scale run |
| `speed.json` | per-structure timings |
| `benchmark_vs_ase.json` | agreement with the reference algorithms |
| `jac_panels.json` | polyhedra, cell edges and lattice for the four Figure 1 structures |

### figures

`figdata.json` (Figure 3), `screen_summary.json` (Figure 5), `spectra_curves.json` and
`spectra_descriptors.json` (Figure 6) hold the plotted values. Figure 4 is drawn from
`data/validation_allmetal703.json` and `data/baseline_scale.json` directly.

## Reproducing the paper's numbers

Requires Python 3.11 with `pymatgen` (2026.5.18) and `numpy`; `ase` (3.26.0) only for the
whole-network comparison. Structures are not redistributed here; they are named by
Crystallography Open Database entry number and read from a local cache, given below as
`<cache>`.

**Complete the cache first.** A snapshot of the COD bulk archive is not sufficient on its
own: entries deposited after the snapshot was taken are absent from it, and an incomplete
cache silently indexes fewer structures and reports different agreement figures. Of the 703
halides, 58 are missing from a bulk archive downloaded in 2025. Run

    python code/fetch_cifs.py data/validation_guards.json <cache> --check
    python code/fetch_cifs.py data/validation_guards.json <cache>

until it reports the cache complete, and only then run what follows.

    python code/harvest_validation.py       <cache> validation_set.json
    python code/reindex_multihalide.py      validation_set.json <cache> validation_multihalide.json
    python code/reindex_allmetal.py         data/validation_guards.json validation_allmetal703.json <cache>
    python code/sweep_rnext.py              data/validation_guards.json sweep_rnext.json <cache>
    python code/frac_stratified.py          data/validation_allmetal703.json
    python code/baseline_and_scale.py       validation_multihalide.json <cache> baseline_scale.json
    python code/benchmark_speed.py          <cache> speed.json

`reindex_allmetal.py` reproduces 397 of 444 framework halides (89.4 per cent), 524 of 703
overall (74.5 per cent) and 127 of 259 for the copper, silver, cadmium and manganese
halides (49.0 per cent), which are the figures reported in Section 3. If it reports fewer
than 703 structures indexed, the cache is incomplete and none of the three figures will
match; go back to `fetch_cifs.py`.

## Limits

Cells containing more than 90 metal centres are declined rather than evaluated. No
structure in the 703-halide validation set reaches that limit; the largest presents 64
metal centres. The index describes the sublattice it is given and makes no claim about
the rest of the structure — that restriction is the point of the method, not a
shortcoming of it.

## Licence

Code under the MIT licence; data under CC BY 4.0.
