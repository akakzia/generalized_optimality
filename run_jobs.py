import os


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']

# Make top level directories
mkdir_p(job_directory)

gammas = [0.71 * 0.01*i for i in range(15)]

for gamma1 in gammas:
    for gamma2 in gammas:
        job_file = os.path.join(job_directory, "{}_{}.job".format(gamma1, gamma2))

        with open(job_file) as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --account=oke@cpu\n")
            fh.writelines("# SBATCH --job-name=test\n")
            fh.writelines("# SBATCH --partition=cpu_p1\n")
            fh.writelines("# SBATCH --qos=qos_cpu-dev\n")
            fh.writelines("# SBATCH --output=gsac_{}_{}_%j.out\n".format(gamma1, gamma2))
            fh.writelines("# SBATCH --error=gsac_{}_{}_%j.out\n".format(gamma1, gamma2))
            fh.writelines("# SBATCH --time=1:00:00\n")
            fh.writelines("# SBATCH --nodes=1\n")
            fh.writelines("# SBATCH --ntasks=1\n")
            fh.writelines("# SBATCH --hint=nomultithread\n")

            fh.writelines("module load pytorch-cpu/py3/1.4.0")

            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include\n")
            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uqy56ga/.mujoco/mujoco200/bin\n")

            fh.writelines("srun python -u -B main.py --agent 'GSAC' --gamma-1 {} --gamma-2 {} --save-dir 'experiments' 2>&1".format(gamma1, gamma2))

        os.system("sbatch %s" % job_file)