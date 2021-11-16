# Python standard library
from typing import *

def histplot_df(df, cols: List[str], aspect_ratio: str = 'wide', bins: Union[int, str] = 'auto', discrete: bool = False, hue: str = None, hue_order: List[str] = None):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set(
        context="talk",
        font_scale=1,  # make the font larger; default is pretty small
        style="ticks",  # make the background white with black lines
        palette="colorblind",  # a color palette that is colorblind friendly!
    )

    # Gets square(ish) dimensions for the number of subplots to make
    # Still knows to, for example, generate a 1x2 grid for 2 plots, rather than a 2x2 grid
    num_plots = len(cols)
    subplot_cols = int(np.ceil(np.sqrt(num_plots)))
    subplot_rows = int(np.ceil(num_plots / subplot_cols))

    if aspect_ratio != 'square':
        # Adjusts the aspect ratio to decrease the number of unused subplots. For example, turns the 3x3 grid generated
        # for 8 subplots into a 2x4 grid.
        # Doesn't deviate too far from square dimensions. For example, turns the 4x4 grid generated for 13 subplots into
        # a 3x5 grid with one fewer empty subplot, but doesn't go all the way to a 1x13 or a 2x6 grid. I actually like this
        # behavior; a 1x13 grid would be annoying to look at.
        sum_dims = subplot_cols + subplot_rows
        short_dim = int(np.ceil((sum_dims - np.sqrt(np.square(sum_dims)-4*num_plots)) / 2))
        long_dim = sum_dims - short_dim
        if aspect_ratio == 'wide':
            subplot_rows = short_dim
            subplot_cols = long_dim
        elif aspect_ratio == 'tall':
            subplot_rows = long_dim
            subplot_cols = short_dim
        else:
            raise ValueError('Must provide a valid aspect ratio')

    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(subplot_cols*4, subplot_rows*4))

    for ax, col in zip(axs.flatten(),cols):
        sns.histplot(data=df, x=col, ax=ax, bins=bins, discrete=discrete, hue=hue, hue_order=hue_order)

    fig.tight_layout()