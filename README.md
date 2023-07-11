# atomlib: A modern, extensible library for creating atomic structures

`atomlib` is a package for creating, modifying, and controlling atomic structures. It draws heavy inspiration from previous tools like [Atomsk][atomsk] and [ASE][ase], but attempts to provide a cleaner, more consistent interface that can be used from Python or a command line.

`atomlib` has minimal dependencies: `numpy`, `scipy`, and `polars` are required for core atom structure manipulation, and `click` is required for command line functionality.

## Atomic representation & supported properties

Atomic structures are stored as `polars` DataFrames, providing a clean, immutable interface that maximizes expressiveness and minimizes errors.
For formats that allow arbitrary properties, these properties can be passed through transparently. `atomlib` has first-class support for fractional occupancy, Debye-Waller factors, atomic forces, and labels.

Translational symmetry is stored in `Cell` objects, which represent a fully generic cell. Atoms can be modified in any coordinate system that makes sense (global, local real-space, cell fraction, box fraction, etc.). Support for non-translational symmetry operations is limited at this point.

For more information, check out the example notebooks and the API documentation.

## Currently supported file formats

File format support is still a work in progress. Where possible, parsers are implemented from scratch in-repo.
Most formats are implemented in two steps: Parsing to an intermediate representation which preserves all format-specific information, and then conversion to the generic `Atoms` & `AtomCell` types for manipulation & display.
This means you can write your own code to utilize advanced format features even if they're not supported out of the box.

| Format                 | Ext.    | Read               | Write              | Notes |
| :--------------------- | :------ | :----------------: | :----------------: | :---- |
| [CIF][cif]             | .cif    | :white_check_mark: | :x:                | CIF1 & CIF2. Isotropic B-factor only |
| [XCrysDen][xsf]        | .xsf    | :white_check_mark: | :white_check_mark: |       |
| [AtomEye CFG][cfg]     | .cfg    | :white_check_mark: | :white_check_mark: | Currently basic format only |
| [Basic XYZ][xyz]       | .xyz    | :white_check_mark: | :white_check_mark: |       |
| [Ext. XYZ][xyz]        | .exyz   | :white_check_mark: | :white_check_mark: | Arbitrary properties not implemented |
| [Special XYZ][xyz]     | .sxyz   | :x:                | :x:                | To be implemented |
| [LAMMPS Data][lmp]     | .lmp    | :x:                | :white_check_mark: |       |
| [Quantum Espresso][qe] | .qe     | :x:                | :white_check_mark: | pw.x format  |
| [pyMultislicer][pyM]   | .mslice | :x:                | :white_check_mark: |       |

[atomsk]: https://atomsk.univ-lille.fr/
[ase]: https://wiki.fysik.dtu.dk/ase/
[cif]: https://www.iucr.org/resources/cif
[xsf]: http://www.xcrysden.org/doc/XSF.html
[cfg]: https://atomsk.univ-lille.fr/doc/en/format_cfg.html
[xyz]: https://atomsk.univ-lille.fr/doc/en/format_xyz.html
[lmp]: https://docs.lammps.org/read_data.html
[qe]: https://www.quantum-espresso.org/Doc/INPUT_PW.html
[pyM]: https://github.com/LeBeauGroup/pyMultislicer
