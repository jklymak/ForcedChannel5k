# Some Runs

## Runs:

  - ChannelToy01: 20x20 km, wind forced, temp restore.  visc=5e-4, diffz=1e-5
  - ChannelToy02: diffz=5e-4; should be much more mixing downwards
    - result is not much different than above!
  - ChannelToy03: f=-1.263e-4, diffz=5e-4;  
    - branch: toy-version
    - this one is not much different either.  maybe a bit better deep strat.
    - This should prob be my base just to f is correct...
  - ChannelToyLowTau: weaker forcing
  - ChannelToyHighdT: Much bigger surface density difference (24 deg!)
  - ChannelToyHighCd: ChannelToy03 at 100 h.  Restart Cd=0.021 (from 0.0021)
    - See how long it takes to settle down.  



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
