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
    - `Channel5k1000_rough_01`: keep running smooth from that spinup.  : `gendataSpunup.py`  starts with
    `Channel5k1000_01` at 5y of run time, and runs for another x years.  Run
    with `source runSpunup.sh Channel5k1000_rough_01`

    - Running `Channel5k1000_rough_01` with fast save time on levels: `fastlevels.nc`  `../results/Channel5k1000_rough_01/input/fastlevels.nc`
    got 52 h.  

    - Running `channel5k1000_02` with longer forcing timescale.  WIll need spunup version as well, but after we see if it makes a difference to the eddy field per Dhruv's email.  

    - Running `channel5k1000_vrough_01` but rough all through channel (no envelope)

    - Running: `channel5k1000_lindrag_01` with smooth bathy but linear drag derived from my parameterization.  Note that it gives a similar drag (2.3e-3 m/s) to those suggested by Marshall et al as being "large".

## Todo:

   - 2-D fft of `fastlevels.nc`: See python/PlotPowerSpectra.ipynb

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
