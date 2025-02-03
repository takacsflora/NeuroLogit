
#!/bin/bash -l


# array job style script

#$ -N AV_neurons_linear
#$ -l h_rt=12:00:00
#$ -l mem=8G

# 11 dataset,15 models/set
#$ -t 1-1

module purge
module load gcc-libs/4.9.2
module load python3/recommended
module load compilers/gnu/4.9.2
module load numactl/2.0.12
module load binutils/2.29.1/gnu-4.9.2
module load ucx/1.9.0/gnu-4.9.2
module load mpi/openmpi/4.1.1/gnu-4.9.2
module load mpi4py/3.1.4/gnu-4.9.2

module load numactl
module load binutils

cd NeuroLogit
pip install --upgrade pip
pip install psutil 
pip install .

python src/batch/cluster_run/fit_encoding_nparray.py $SGE_TASK_ID

deactivate