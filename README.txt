1. writing your script (e.g. "pyhon_script.py") //local computer
2. writing slurm file (e.g. "submit_file.slurm") //local computer
3. creating new dir (preferably by the job name e.g "python_script") and save both files in it //slurm server and filezilla
4. enter <job_name> dir //slum server
5. run in terminal: bash //slurm server
6. run in terminal: sbatch submit_file.slurm //slurm server
7. wait for <job_name>.out and <job_name>.err