# Some Runs



## Runs:


## To setup

  - need a 20km run from Hornby.  Files like `'ChannelToy03Last.nc'` and the
  2-d verions.  These are the spinup for these runs;  Done in
  `input/get20kFile.py` and stored in `input/Channel1000Spinup2d.nc`  and `input/Channel1000Spinup.nc`.  
  - Run 5k for 10 y:
    - eddies obviously matter.
    - run 5k rough for 10 y from that spinup.  Rough should be made from 1km so
    we can reuse later.  
    - keep running smooth from that spinup.  : `gendataSpunup.py`  


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
