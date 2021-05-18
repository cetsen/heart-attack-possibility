import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
    
    
def plot_dists(df):
    ''' 
    Plots the histogram of each column (including the target) on a 4x4 subplot
    
    Args:
        df (pandas DataFrame):  dataframe including one column per feature and target
    Returns: 
        None
    '''
    fig, ax = plt.subplots(4,4, figsize=(12,8))
    
    for i, column_name in enumerate(df.columns):
        sns.histplot(data=df, x=column_name, ax=ax[i//4,i%4])
        
    plt.suptitle('Distributions of features and target')
    
    # Space adjustment for title
    fig.subplots_adjust(top=0.88)
    
    fig.tight_layout()
    plt.show()
        

def plot_feat_target(df):
    '''
    Function taken from CS-421 'Machine Learning for Behavioral Data' course homework solutions at EPFL
    
    Plots the distribution of each feature based on the target
    
    Args:
        df (pandas DataFrame): dataframe including one column per feature and the target column 'target'
    Returns: 
        None
    '''
       
    # Identify and group features based on their type
    continuous_cols = [col for col in df._get_numeric_data().columns if df[col].nunique() > 5]
    categorical_cols = list(set(df.columns) - set(continuous_cols))

    # Create a plot grid of shape (n_features // 2, 2), with one plot per feature
    fig, axes = plt.subplots(len(df.columns)//2, 2, figsize=(14,14))
    
    # For each feature in the dataframe df, create a plot to relate the values of that feature with the target 
    for i, col in enumerate(df.drop('target', axis=1).columns):   
        
        # Select the axis the plot will be added
        ax = axes[i // 2, i % 2]
            
        # For each unique feature value, count the percentage of target=1 and target=0
        col_counts = df.groupby(by=col)['target'].value_counts(normalize=True).rename('Percentage').mul(100)
        col_counts = col_counts.reset_index().sort_values(col)                       
        
        # Show a bar plot if the feature is categorical, a histogram otherwise
        if col in categorical_cols:  
            g = sns.barplot(data=col_counts, x=col, y='Percentage', hue='target', ax=ax)
            g.legend(loc='upper right', ncol=2)
            g.set(ylim=(0., 100.))
        else:
            g = sns.kdeplot(data=df, x=col, hue='target', ax=ax)
            g.set(xlim=(0., None))
            
    # In case the number of feature is even, remove the last empty plot
    if (len(df.columns) - 1) % 2 != 0:
        fig.delaxes(axes[len(df.columns) // 2-1][1])
      
    plt.suptitle('Distribution of each feature based on target')
    fig.tight_layout()
    plt.show()
    
    
def plot_corr_heatmap(df):
    """
    Plots correlation heatmap between each column
    
    Args:
        df (pandas DataFrame): dataframe including one column per feature and target
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(14,8)) 
    plt.title("Correlation Heatmap") 
    
    # Show only lower triangle
    mask = np.triu(df.corr()) 
    
    sns.heatmap(df.corr(), annot = True, ax=ax, mask=mask, cmap='coolwarm', linewidths=.5) 
    
    
def plot_model_comparison(df):
    
