import matplotlib.pyplot as plt
import seaborn as sns


def my_histplot(df,
                title="Распределение",
                xlabel="Значение",
                ylabel="Частота",
                bins=50,
                figsize=(4, 4),
):
    """
    Построение гистограммы
    """
    plt.figure(figsize=figsize)
    sns.histplot(df, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def my_boxplot(df, 
               title="Boxplot", 
               xlabel="Значение", 
               figsize=(4, 4)
):
    """
    Построение boxplot
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(True, alpha=0.3)
    plt.show()
