# Probing bulk water properties with equivariant NNPs

Final ML project for CHEM 31300 - Structural and dynamical properties of liquid water using invariant/equivariant neural network potentials

## Overview

This project compares two machine-learned interatomic potentials on the same liquid water system:

| Model | Type | Potential file | Engine |
|---|---|---|---|
| **DeePMD (MB-pol)** | Invariant descriptor-based NNP (Deep Potential) trained on MB-pol reference data, from the Paesani group (UC San Diego) | `deepmd_mbpol/graph-compress.pb` (compressed frozen graph) | LAMMPS + DeePMD-kit (`pair_style deepmd`), CPU/MPI |
| **MACE (MATPES-r2SCAN)** | E(3)-equivariant message-passing NNP; foundation model fine-tuned on r²SCAN data | `MACE-matpes-r2scan-omat-ft.model-lammps.pt` (not included; see below) | LAMMPS + MACE (`pair_style mace`), GPU via Kokkos |

Both models run NVT molecular dynamics of bulk water at 300 K. The trajectories are then analyzed for:

- **Structure**: three-body angular distribution functions (ADFs), for example the O–O–O angle distribution within the first solvation shell, which measures tetrahedral order (`adf.py`).
- **Dynamics**: mean squared displacement (MSD), from which you can estimate the self-diffusion coefficient (`msd.py`).

## Repository structure

```
.
├── README.md
├── adf.py                          # Three-body angular distribution function (J–I–K triplets)
├── msd.py                          # Mean squared displacement from a LAMMPS trajectory
├── deepmd_mbpol/
│   ├── in.water                    # LAMMPS input: DeePMD NVT run
│   ├── data.water_AA_dp_difflabel  # Initial configuration (O = type 1, H = type 2)
│   ├── graph-compress.pb           # Compressed DeePMD model (~37 MB)
│   └── lammps.sh                   # SLURM job script (48 MPI ranks, CPU)
└── mace/
    ├── water.in                    # LAMMPS input: MACE NVT run
    ├── data.water_AA_dp            # Initial configuration (H = type 1, O = type 2)
    └── script_gpu.sh               # SLURM job script (1 GPU, Kokkos)
```

## System

Both simulations start from the same configuration:

| Property | Value |
|---|---|
| Molecules | 512 H₂O (1536 atoms) |
| Box | Cubic, 25.0222 Å per side, periodic in x, y and z |
| Density | ≈ 0.978 g/cm³ |
| LAMMPS units | `metal` (Å, ps, eV) |
| `atom_style` | `atomic` (no explicit bonds; the NNP handles all interactions) |

> **Atom type ordering differs between the two data files.** The DeePMD file uses O = 1, H = 2. The MACE file uses H = 1, O = 2, matching the `pair_coeff * * <model> H O` element mapping. Keep this in mind when selecting atom types in the analysis scripts.

## Simulation details

| Setting | DeePMD (`deepmd_mbpol/in.water`) | MACE (`mace/water.in`) |
|---|---|---|
| Ensemble | NVT (Nosé–Hoover), 300 K, `Tdamp` = 0.1 ps | NVT (Nosé–Hoover), 300 K, `Tdamp` = 0.1 ps |
| Timestep | 1 fs | 1 fs |
| Run length | 20,000,000 steps (20 ns)\* | 1,000,000 steps (1 ns) |
| Initial velocities | Gaussian at 300 K, random seed passed via `-var SEED` | Gaussian at 300 K, seed 61231, zero net momentum |
| Trajectory output | `water_aa_nn.lammpstrj` every 500 steps (0.5 ps): `id type x y z fx fy fz vx vy vz` | `dump.lammpstrj` every 500 steps (0.5 ps): `id type x y z` |
| Restarts | Every 5000 steps → `water_cg.restart{1,2}` | Every 1000 steps → `config.restart{1,2}` |
| Thermo | Every 2000 steps, `multi` style | Every 10 steps, with energies, box lengths and virial/pair pressure tensor components |
| Final output | — | `all_traj.lammpstrj`, `water.data` |

\* The comment in `in.water` says "10 ns", but `run 20000000` at a 1 fs timestep is 20 ns. Adjust whichever is wrong.

## Requirements

### Simulations

- **DeePMD run**: LAMMPS built with the [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) plugin (`lmp_mpi`, providing `pair_style deepmd`) and an MPI implementation. The job script also points to a PLUMED kernel, but the input does not use PLUMED.
- **MACE run**: LAMMPS built with [MACE](https://github.com/ACEsuit/mace) support (`pair_style mace`) and Kokkos/CUDA, plus a CUDA-capable GPU. You also need the MACE-MATPES-r2SCAN foundation model compiled for LAMMPS (`MACE-matpes-r2scan-omat-ft.model-lammps.pt`). It is not in this repo; download it from the MACE foundation-models release and convert it with `mace_create_lammps_model`.

### Analysis

- Python 3
- `numpy`
- `numba` (JIT-compiles the triple loop in `adf.py`)
- `matplotlib`

```bash
pip install numpy numba matplotlib
```

## Running the simulations

The job scripts are SLURM templates. Before running, edit the following in each script:

1. **SLURM details:** add your account, partition, wall time and so on at the line marked `# >>> Insert SLURM details here`. The resource requests (`--nodes`, `--ntasks-per-node`, `--gres=gpu:1`) are already set but can be changed.
2. **Modules:** load the modules your cluster needs (MPI, CUDA, etc.) at the line marked `# >>> Insert module loads for your cluster here`.
3. **Paths:** replace the following placeholders with your local paths:

| Placeholder | File | What it should point to |
|---|---|---|
| `<PATH_TO_DEEPMD_CONDA_ENV>` | `deepmd_mbpol/lammps.sh` | Conda environment with DeePMD-kit and the `lmp_mpi` binary |
| `<PATH_TO_PLUMED_INSTALL>` | `deepmd_mbpol/lammps.sh` | PLUMED install prefix (containing `lib/libplumedKernel.so`); optional, since the input doesn't use PLUMED |
| `<PATH_TO_MACE_CONDA_ENV>` | `mace/script_gpu.sh` | Conda environment with MACE / PyTorch |
| `<PATH_TO_MACE_LAMMPS_BUILD>` | `mace/script_gpu.sh` | LAMMPS build directory containing the MACE-enabled `lmp` binary |
| `<PATH_TO_MACE_MODEL_DIR>` | `mace/water.in` | Directory containing `MACE-matpes-r2scan-omat-ft.model-lammps.pt` |

The scripts won't run until you replace these placeholders. The shell treats `<...>` as a redirection, so it fails immediately instead of running with a wrong path.

### DeePMD (MB-pol)

```bash
cd deepmd_mbpol
sbatch lammps.sh
# or, without SLURM:
mpirun -np 48 lmp_mpi -in in.water -var SEED $RANDOM
```

`in.water` requires the `SEED` variable. LAMMPS stops with an error if you don't pass it.

### MACE

```bash
cd mace
# replace <PATH_TO_MACE_MODEL_DIR> in water.in with your local model directory
sbatch script_gpu.sh
# or, without SLURM:
lmp -k on g 1 -sf kk -in water.in
```

`pair_style mace no_domain_decomposition` is meant for a single GPU/MPI rank.

## Analysis scripts

Both scripts read a LAMMPS custom-format trajectory from the hard-coded path `../mapping/com.lammpstrj`. This is a trajectory mapped to molecular centers of mass (one site per water molecule), produced by a separate mapping step that is not in this repository. To analyze a raw simulation trajectory instead, edit the file path near the top of each script, for example to `water_aa_nn.lammpstrj` or `dump.lammpstrj`.

### `adf.py`: angular distribution function

Computes the distribution of angles θ<sub>JIK</sub> for every triplet where a central site of type I has two neighbors, of types J and K, both closer than a cutoff `sigma`. Periodic boundaries are handled with the minimum-image convention.

**Parameters** (set in `main()`):

| Parameter | Default | Meaning |
|---|---|---|
| `target_j, target_i, target_k` | `1, 1, 1` | Atom types of the J–I–K triplet (I is the central atom) |
| `sigma` | 3.7 Å | Neighbor cutoff (roughly the first minimum of g(r) for O–O / COM–COM) |
| `dang` | 0.5° | Angular bin width |
| `angmin, angmax` | 0°, 180° | Angular range |

**Expected trajectory format:** exactly 9 header lines per frame, a fixed number of atoms, and per-atom columns `id type x y z fx fy fz` in that order. The script reads the box from the first frame only, so it assumes a constant volume (fine for NVT).

**Outputs:**
- `adf_<J><I><K>.dat`: columns are angle (deg), angle (rad), normalized P(θ) (integrates to 1 over degrees)
- `adf_plot.png`: plot of P(θ)

```bash
python adf.py
```

For the O–O–O distribution on the raw DeePMD trajectory, use types `1, 1, 1`. On the raw MACE trajectory, oxygen is type `2`. The MACE dump doesn't write forces, so either add `fx fy fz` to the dump or remove the force-parsing lines from `adf.py`.

The loop is O(N_I · N_J · N_K) per frame. It runs quickly on a 512-site COM trajectory but slows down a lot on the full 1536-atom system.

### `msd.py`: mean squared displacement

Reads the trajectory frame by frame, matches atoms by `id`, and computes

MSD(t) = ⟨ |**r**<sub>i</sub>(t) − **r**<sub>i</sub>(0)|² ⟩

averaged over all sites, with the first frame as the single time origin. The script converts timestep numbers to ps by multiplying by 0.001, which assumes a 1 fs timestep.

**Output:** `msd_deep_mbpol.dat` with columns `Time(ps) MSD(Å²)`. Rename the output file when analyzing the MACE trajectory.

```bash
python msd.py
```

You can get the self-diffusion coefficient from the slope of the linear (diffusive) regime: D = (1/6) · d⟨Δr²⟩/dt. Note that 1 Å²/ps = 10⁻⁴ cm²/s.

## Known limitations

- **MSD and periodic boundaries:** `msd.py` applies the minimum-image convention to the total displacement from the reference frame. This is only correct while every displacement stays below half the box length (~12.5 Å). At longer times the MSD artificially plateaus near L²/4. For long trajectories, use unwrapped coordinates (dump `xu yu zu`, or `ix iy iz` image flags) and remove the PBC correction.
- **Single time origin:** `msd.py` uses only the first frame as the origin. Averaging over multiple time origins would give much better statistics.
- **Hard-coded paths and parameters:** the input trajectory path, atom types and output filenames are set inside the scripts. Edit them for each analysis.
- **Finite-size effects:** diffusion coefficients from a 512-molecule periodic box are lower than the infinite-system value. Apply the Yeh–Hummer correction when comparing to experiment (~2.3 × 10⁻⁵ cm²/s at 298 K).

## Author

Kuntal Ghosh, CHEM 31300 (Machine Learning in Chemistry), University of Chicago
