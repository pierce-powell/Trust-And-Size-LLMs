import pandas as pd
import os

def initial_trust_avg(df):
    '''
    calculate the initial trust (aka the average trust on the first round of each game) for each family/size
    '''
    # Filter for the first round of each game
    first_round_df = df[df['round'] == 1]

    # Group by model and variant, then calculate the mean of 'model_choice' for each group
    initial_trust_df = first_round_df.groupby(['model', 'variant', 'not_gamified'])['model_choice'].apply(lambda x: (x == 'C').mean()).reset_index()

    # Print the initial trust dataframe
    print(initial_trust_df)

    # Print the count of each group to verify the number of entries for each model and variant
    print(first_round_df.groupby(['model', 'variant', 'not_gamified']).size().reset_index(name='count'))

    return initial_trust_df.rename(columns={'model_choice': 'avg_initial_trust'})



if __name__ == "__main__":
    
    raw_path = r"C:\Users\Sonic\OneDrive\Desktop\git-repositories\LLMs\Trust-And-Size-LLMs\Raw Data"

    input_path = os.path.join(raw_path, r"DeepSeek IPD\cleaned_deepseek_33B_IPD.removed_rows.csv")
    output_path = os.path.join(raw_path, "initial_trust.csv")
    
    df = pd.read_csv(input_path)[['model', 'variant', 'round', 'model_choice', 'not_gamified']]
    initial_trust_df = initial_trust_avg(df)
        
    initial_trust_df.to_csv(
        output_path,
        mode='a',                      # append
        header=not os.path.exists(output_path),  # only write header once
        index=False
    )
