#!/bin/bash
#
#SBATCH --job-name=ttcf
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=128
#SBATCH --array=1-1000
 module load gcc/8.2.0
 module load openmpi/4.0.0
 module load lammps/20200303

seed_file=seed.txt
Child_No='100'
Shear_rate='0.01'


mkdir batch$SLURM_ARRAY_TASK_ID
cp mother.in batch$SLURM_ARRAY_TASK_ID/mother.in
cp child.in batch$SLURM_ARRAY_TASK_ID/child.in
cp slab.lmp batch$SLURM_ARRAY_TASK_ID/slab.lmp
cp vel.lmp batch$SLURM_ARRAY_TASK_ID/vel.lmp
cp force.lmp batch$SLURM_ARRAY_TASK_ID/force.lmp
cp force_equil.lmp batch$SLURM_ARRAY_TASK_ID/force_equil.lmp
cp sim_params.lmp batch$SLURM_ARRAY_TASK_ID/sim_params.lmp
cp ${seed_file} batch$SLURM_ARRAY_TASK_ID/${seed_file}


cd batch$SLURM_ARRAY_TASK_ID

seed_v=`sed -n ${SLURM_ARRAY_TASK_ID}p ${seed_file}`

mpirun -np 1 lmp -screen none -var rseed ${seed_v} -var Nchild "${Child_No}" -in mother.in 


for ((i=1;i<="${Child_No}";i++))
do

mv branch$i.dat branch.dat

for ((j=1;j<=4;j++))
do

mpirun -np 1 lmp -screen none -var map "${j}" -var child_index "${i}" -var srate "${Shear_rate}" -in child.in

done
done

