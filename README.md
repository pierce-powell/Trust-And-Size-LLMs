# Steps for running things
First clone the repository
These reproducability steps are assuming you will be running on your own system. 

1. **Download models on your local machine**

2. **Set up your virtual environment** by installing the required dependencies.

3. **Edit `run_ipd.slurm`**:
   - a. Change  
     `source ~/miniconda3/etc/profile.d/conda.sh`
     `source ~/Trust/bin/activate`  
     to use the virtual environment you just configured.
   - b. In  
     `accelerate launch run_ipd.py --out raw_qwen_0.5B_IPD.csv --rounds 100 --variants default,game-theorist`  
     change the `.csv` filename to match your test.
   - c. Change  
     `ACCELERATE_CONFIG_FILE=/home/XXX/Trust-And-Size-LLMs/default_config.yaml`  
     to use your username.
   - d. Update the model path and model name to match your file setup.

4. **Run everything** with:
```bash
   sbatch run_ipd.slurm
```

5. Repeat this for every model you wish to gather statistics for. If you're job dies mid run, the code will read the csv and pick up where it left off the last time.

6. Make sure to run `cleanup_fallbacks.py` for each csv created to get the corrected statistics.

# Graphing scripts are also provided in the repo: 
Each has instructions on how to run the corresponding graph at the top of their files, the graphing scrits are as follows:
1. compare_results.py
2. comare_results_all.py
3. Trust_Recovery.py
4. interaction_plot.py

# Note
We no longer use the Dictator game in our experiment, so the lines of code pertaining to Dictator can be ignored.