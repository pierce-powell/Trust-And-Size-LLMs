 "# Initial Setup" 
1. Setup or modify one of the config files (i.e. test_ipd.yaml) to include the experimental conditions you want (models, prompts, rounds, etc.)
2. Using that new yaml config you made/edited, download the models to your local machine 
    python download_models_local.py --config pipeline/test_ipd.yaml --cache_dir ./hf_cache 
3. Zip that file
    tar -czvf hf_cache.tar.gz hf_cache/
4. From stokes/newton or whatever computing cluster you are using setup the cache 
    mkdir -p ~/hf_home/transformers
    tar -xzvf hf_cache.tar.gz -C ~/hf_home/transformers
    export TRANSFORMERS_CACHE=/home/pi047867/hf_home/transformers



#Other useful commands 
Convert to Unix line endings: 
dos2unix [file_name].slurm


#Exporting HF env variables: 
export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HOME/hf_cache
