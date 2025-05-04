# Get results of evaluation

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

# Load classnames

with open(os.path.expanduser("~/project/geneval/prompts/object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results
filenames = os.listdir(args.filename)

def get_single_df(filename):
    df = pd.read_json(filename, orient="records", lines=True)

    # Measure overall success

    print("Summary")
    print("=======")
    print(f"Total images: {len(df)}")
    print(f"Total prompts: {len(df.groupby('metadata'))}")
    print(f"% correct images: {df['correct'].mean():.2%}")
    print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
    print()

    # By group

    task_scores = []

    print("Task breakdown")
    print("==============")
    for tag, task_df in df.groupby('tag', sort=False):
        task_scores.append(task_df['correct'].mean())
        print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
    print()

    print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")
    return df
    
all_dfs = []
for filename in filenames:
    df = pd.read_json(os.path.join(args.filename, filename), orient="records", lines=True)
    df['filename'] = filename
    all_dfs.append(df)
# Concatenate all dataframes
all_df = pd.concat(all_dfs, ignore_index=True)

tags = all_df['tag'].unique().tolist()
mean_score_df = pd.DataFrame(columns=['filename'] + tags + ['mean_score'], index=range(len(all_df['filename'].unique())))
for i, (filename, group) in enumerate(all_df.groupby('filename')):
    mean_score = group['correct'].mean()
    mean_score_df.iloc[i] = {'filename': filename, **{tag: group[group['tag'] == tag]['correct'].mean() for tag in tags}, 'mean_score': mean_score}

mean_score_df.to_csv('mean_scores.csv', index=False)
    

    