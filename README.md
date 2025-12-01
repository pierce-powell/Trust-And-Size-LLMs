# Steps for running things
These reproducability steps are assuming you will be running things on a job based system. If you are not, you can simply just use the contents of the slurm yourself. 

1. **Download models on your local machine** and then upload them to the remote server through an FTP client.  
   (You can also download the models locally by running `download_local.py` and editing the model names.)

2. **Set up your virtual environment** by installing the required dependencies.

3. **Edit `run_ipd.slurm` and `run_dicator.slurm`**:
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
and 
```bash
   sbatch run_dictator.slurm
```

5. Repeat this for every model you wish to gather statistics for. If you're job dies mid run, the code will read the csv and pick up where it left off the last time. 

6. There was an error with how cooperation probability was calculated mid run, as well as a small error with writting duplicate values to the csv during the runs. So once your data is finished collecting, use the clean up script to clean up both the dictator results and the ipd results: 


```bash
   python cleanup_fallbacks.py --infile input.csv --outfile output.csv
```

# Graphing scripts are also provided in the repo: 
Each has instructions on how to run the corresponding graph at the top of their files, the graphing scrits are as follows: 
1. compare_results.py
2. compare_results_dict.py
3. comare_results_between.py
4. comare_results_between_dict.py
5. Trust_Recovery.py 


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