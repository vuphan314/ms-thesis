#!/bin/bash

################################################################################

## /scratch, 1000s

################################################################################

#SBATCH --time=0-3:0:0 # days-hours:minutes:seconds
  ## 14h = 50400s > 50000s = 100s * 500
  ## 3h = 10800s > 10000s = 10s * 1000 = 1000s * 10
#SBATCH --partition=scavenge # commons 1d; scavenge 4h; interactive 30m

#SBATCH --mem=24G
#SBATCH --constraint=ivybridge
  ## ivybridge
    ## nodes     136
    ## CPU       2.6 GHz Xeon E5-2650v2
    ## RAM (GB)  32-128
  ## broadwell
    ## nodes     28
    ## CPU       2.2 GHz Xeon E5-2650v4
    ## RAM (GB)  32-128
  ## skylake
    ## nodes     60
    ## CPU       2.6 GHz Xeon Gold 6126
    ## RAM (GB)  192

#SBATCH --nodes=1 --exclusive

#SBATCH --mail-type=ALL # uses my school email if no address is specified

#SBATCH --profile=ALL

#SBATCH --job-name=ADDMC_experiment

################################################################################

module load GCC/6.4.0 OpenMPI/2.1.3 Git/2.14.1 Singularity/2.5.2 Python/3.6.4

. ${countingScriptsPath}/c${SLURM_ARRAY_TASK_ID}.sh # ${countingScriptsPath} will be exported by experiment.py
