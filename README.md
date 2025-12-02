# 5AT020-eDrives
CBL Assignment eDrives

```
/5AT020-eDrives
├── /weights            # Precomputed weights for DDPG implemetation
│
├── /images             # Images to show in jupyter notebook
│
├── controllers.py      # PI and MPC implementation
├── job.slurm           # Example of bash file to run in Snellius using SLURM scheduler for jobs (it works with python files, not notebooks)
├── environment.py      # Implementation of the PMSM model using a Gymnasium template (Farama Foundation)
├── pmsm_control.ipynb  # Incomplete example to run PI, MPC, and DDPG
├── requirements.txt    # Required packages to run 'pmsm_control.ipynb'
├── test_slurm.py       # Example of python file to test 'job.slurm'
├── utils.py            # Complementary functions to run 'pmsm_control.ipynb'
└── README.md            
```

To run a SLURM job use `sbatch` following the next example:
```console
user@hostname:~/$ sbatch job.slurm
```
For more information on the command usage run `sbatch --help`.
