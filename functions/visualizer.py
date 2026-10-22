import matplotlib.pyplot as plt
import seaborn as sns

def plot_count_pairs(df, col, hue=None, title=None):
    """
    Plot count pairs for a given column.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        col (str): The column to plot.
        hue (str, optional): The column to group by (e.g., 'set' or 'Survived').
        title (str, optional): The title of the plot.
    """
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    color_list = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    
    sns.countplot(x=col, data=df, hue=hue, palette=color_list)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    
    if title:
        ax.set_title(title)
    plt.show()

def plot_distribution_pairs(df, col, hue=None, title=None):
    """
    Plot distribution pairs for a given column (using histplot).
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        col (str): The column to plot.
        hue (str, optional): The column to group by.
        title (str, optional): The title of the plot.
    """
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    if hue and df[hue].nunique() <= 4:
        # 参考原代码中 Survived (0, 1) 的画法
        colors = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
        for i, h in enumerate(df[hue].unique()):
            sns.histplot(df.loc[df[hue] == h, col], color=colors[i % len(colors)], ax=ax, label=h)
        ax.legend()
    else:
        sns.histplot(df[col], ax=ax)
        
    if title:
        ax.set_title(title)
    plt.show()