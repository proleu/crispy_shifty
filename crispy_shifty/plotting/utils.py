# Python standard library
from typing import *

def histplot_df(df, cols: List[str], bins: Union[int, str] = 'auto', discrete: bool = False, hue: str = None, hue_order: List[str] = None):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set(
        context="talk",
        font_scale=1,  # make the font larger; default is pretty small
        style="ticks",  # make the background white with black lines
        palette="colorblind",  # a color palette that is colorblind friendly!
    )

    num_plots = len(cols)
    subplot_cols = int(np.ceil(np.sqrt(num_plots)))
    subplot_rows = int(np.ceil(num_plots / subplot_cols))

    # Adjust the aspect ratio to minimize the number of unused plots
    sum_rows_cols = subplot_cols + subplot_rows
    subplot_rows = int(np.ceil((sum_rows_cols - np.sqrt(np.square(sum_rows_cols)-4*num_plots)) / 2))
    subplot_cols = sum_rows_cols - subplot_rows

    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(subplot_cols*4, subplot_rows*4))

    for ax, col in zip(axs.flatten(),cols):
        sns.histplot(data=df, x=col, ax=ax, bins=bins, discrete=discrete, hue=hue, hue_order=hue_order)

    fig.tight_layout()