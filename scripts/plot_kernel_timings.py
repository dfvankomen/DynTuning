"""

NOTE: this script is written as if it were run in the root of the project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats

filename = Path("runoutputs/matmul/matmul_N_512_26-07-2024_16-19-11_results.csv")
title_prefix = "Matrix-Matrix Multiplication"
save_prefix = "matmul_image"

# read into pandas
df = pd.read_csv(filename)

# get the unique blocks
dfs = dict(tuple(df.groupby("blocks0")))
calc_regression_lines = []
blks_to_use = None

for blks, data in dfs.items():
    plt.figure()
    plt.scatter(data["threads0"], data["opsTime"])
    plt.grid()
    plt.title(f"Output for {blks} MinBlocks - {title_prefix}")
    plt.savefig(f"{save_prefix}_output_threads_{blks}.png")
    plt.close()

    if (blks == 0):
        continue

    # calculate a regression line
    x = np.array(data["threads0"].to_list())
    y = np.array(data["opsTime"].to_list())
    xidx = np.argsort(x)
    x = x[xidx]
    y = y[xidx]
    if blks_to_use is None:
        blks_to_use = x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    calc_regression_lines.append(blks_to_use * slope + intercept)
    print(blks, "R_value, P_value, std_Err:", r_value, p_value, std_err)

df_split = df.drop(df[df['threads0'] == 0].index)

# rename the blocks option for threads
df_split = df_split.rename(columns={'blocks0': "Min Blocks"})

# Seaborn makes it super easy to use a csv df
plt.figure()
# for ii, x in enumerate(calc_regression_lines):
#     plt.plot(blks_to_use, x, c=sns.color_palette("crest", n_colors=len(calc_regression_lines))[ii], alpha=0.8)
sns.scatterplot(data=df_split, x="threads0", y="opsTime", hue="Min Blocks", size="Min Blocks", palette="crest")
plt.title(f"Execution Time vs Threads - {title_prefix}")
plt.grid()
plt.xlabel("Number of Threads (maxThreads)")
plt.ylabel("Execution Time (s)")

plt.savefig(f"{save_prefix}_output_threads_combined.png")
plt.close()

### SAME FOR THREADS

# get the unique blocks
dfs = dict(tuple(df.groupby("threads0")))
calc_regression_lines = []
blks_to_use = None

for thrds, data in dfs.items():
    plt.figure()
    plt.scatter(data["blocks0"], data["opsTime"])
    plt.grid()
    plt.title(f"Output for {thrds} Threads - {title_prefix}")
    plt.savefig(f"{save_prefix}_output_blocks_{thrds}.png")
    plt.close()

    if (thrds == 0):
        continue

    # calculate a regression line
    x = np.array(data["blocks0"].to_list())
    y = np.array(data["opsTime"].to_list())
    xidx = np.argsort(x)
    x = x[xidx]
    y = y[xidx]
    if blks_to_use is None:
        blks_to_use = x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    calc_regression_lines.append(blks_to_use * slope + intercept)

df_split = df.drop(df[df['blocks0'] == 0].index)

# Seaborn makes it super easy to use a csv df
plt.figure()
for ii, x in enumerate(calc_regression_lines):
    plt.plot(blks_to_use, x, c=sns.color_palette("crest", n_colors=len(calc_regression_lines))[ii], alpha=0.8)
sns.scatterplot(data=df_split, x="blocks0", y="opsTime", hue="threads0", size="threads0", palette="crest")
plt.title(f"Execution Time vs Blocks - {title_prefix}")
plt.grid()
plt.xlabel("Number of Blocks (minBlocks)")
plt.ylabel("Execution Time (s)")
plt.savefig(f"{save_prefix}_output_blocks_combined.png")
plt.close()
