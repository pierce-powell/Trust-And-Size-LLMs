# Steps for running things 
1. Download models on your local machine and then upload them to the remote server through an FTP client 
    (you can also download the models locally that you want by running download_local.py and editing the model names)
2. Once the models are downloaded, go into ipd_test.py and change the model registry to use the models you want to run with 
3. Then set up your venv by downloading the requirements 
4. Once your venv has the necessary requirements, edit run_ipd.slurm 
    a. change "source ~/qwen_env/bin/activate" to use the venv you just configured 
    b. in: accelerate launch ipd_test.py --out results_qwen_32.csv --rounds 100 --variants default,game-theorist,COA
        change the .csv to a name that matches your test 
5. run everything with sbatch run_ipd.slurm 


# Other useful commands
See the status of your job: 
    watch -n 5 squeue -u $USER

Convert the slurm to unix line endings instead of windows 
    dos2unix run_ipd.slurm 

Follow the output of your job live (after the status has shifted from PD to R)
    tail -f ipd_test.[job_id].out


## Pulling the project or cloning it on the remote server
1. Use your private .ssh_stokes dir for keys and known_hosts
mkdir -p ~/.ssh_stokes
touch ~/.ssh_stokes/known_hosts
chmod 700 ~/.ssh_stokes
chmod 600 ~/.ssh_stokes/known_hosts

2. Tell SSH to use that key + known_hosts file
export GIT_SSH_COMMAND='ssh -i /home/[your_user]/.ssh_stokes/id_ed25519_stokes -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/[your_user]/.ssh_stokes/known_hosts'

3. Force Git to use only 1 thread (avoids “unable to create thread”)
git config --global pack.threads "1"

4. Now clone / pull / push works
git clone [repo address]
