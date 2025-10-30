# Steps for running things

1. **Download models on your local machine** and then upload them to the remote server through an FTP client.  
   (You can also download the models locally by running `download_local.py` and editing the model names.)

2. **Set up your virtual environment** by installing the required dependencies.

3. **Edit `run_ipd.slurm`**:
   - a. Change  
     `source ~/qwen_env/bin/activate`  
     to use the virtual environment you just configured.
   - b. In  
     `accelerate launch ipd_test.py --out results_qwen_32.csv --rounds 100 --variants default,game-theorist,COA`  
     change the `.csv` filename to match your test.
   - c. Change  
     `export ACCELERATE_CONFIG_FILE=~/home/pi047867/Trust-And-Size-LLMs`  
     to use your username.
   - d. Update the model path and model name to match your file setup.

4. **Run everything** with:
```bash
   sbatch run_ipd.slurm
```

# Other Useful Commands

**See the status of your job:**
```bash
watch -n 5 squeue -u $USER
```

**Convert the slurm to unix line endings instead of windows** 
```bash
    dos2unix run_ipd.slurm 
```


**Follow the output of your job live (after the status has shifted from PD to R)**
```bash
    tail -f ipd_test.[job_id].out
```

## Pulling the project or cloning it on the remote server
**1. Use your private .ssh_stokes dir for keys and known_hosts**
```bash
mkdir -p ~/.ssh_stokes
touch ~/.ssh_stokes/known_hosts
chmod 700 ~/.ssh_stokes
chmod 600 ~/.ssh_stokes/known_hosts
```

**2. Tell SSH to use that key + known_hosts file**
```bash
export GIT_SSH_COMMAND='ssh -i /home/[your_user]/.ssh_stokes/id_ed25519_stokes -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/[your_user]/.ssh_stokes/known_hosts'
```

**3. Force Git to use only 1 thread (avoids “unable to create thread”)**
```bash
git config --global pack.threads "1"
```

**4. Now clone / pull / push works**
```bash
git clone [repo address]
```