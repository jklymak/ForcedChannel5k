# Some Runs

## Contents:

  - `MITgcm66h` is my version with `NF90io`.
  - `input` is where most model setup occurs.
  - `python` is where most processing occurs.

## Vagaries

   - Need `miniconda3` on the path!

## To compile on Conrad

  - `module load cray-netcdf-hdf5parallel`
  - `cd build/`
  - `../MITgcm66h/tools/genmake2 -optfile=../build_options/conrad -mods=../code/ -rootdir=../MITgcm66h -mpi`
  - `make depend`.  This will have some errors near the end about not being able to find source files for `module netcdf`.  This error is annoying but doesn't affect compile.
  - `make`

## To run

  - run `python gendata.py`
  - run `qsub -N jobname runModel.sh`
