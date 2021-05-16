import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
    
    
def plot_dists(df):
    ''' 
    Plot the distribution of each column
    '''
    fig, ax = plt.subplots(4,4, figsize=(12,8))
    
    for i, column_name in enumerate(df.columns):
        sns.histplot(data=df, x=column_name, ax=ax[i//4,i%4])
        
    plt.suptitle('Distributions of features and target')
    fig.subplots_adjust(top=0.88) # space adjustment for title
    
    fig.tight_layout()
    plt.show()
        

def plot_feat_target(df):
    '''
    For each feature in df, create a plot that shows the percentage of churn (1) or not chrun (0) for each feature value. 
    :param df: A dataframe including one column per feature and the target column 'Churn'.  
    :return: None. 
    '''
       
    # Identify and group features based on their type: numerical and categorical
    continuous_cols = [col for col in df._get_numeric_data().columns if df[col].nunique() > 5]
    categorical_cols = list(set(df.columns) - set(continuous_cols))

    # Create a plot grid of shape (n_features // 2, 2), with one plot per feature
    fig, axes = plt.subplots(len(df.columns)//2, 2, figsize=(14,14))
    
    # For each feature in the dataframe df, create a plot to relate the values of that feature with the 'Churn' target 
    for i, col in enumerate(df.drop('target', axis=1).columns):   
        
        # Select the axis the plot will be added
        ax = axes[i // 2, i % 2]
            
        # For each unique feature value, count the percentage of churn (1) and not chrun (0)
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
      
    plt.suptitle('Distribution of each feature based on target (after cleaning)')
    fig.tight_layout()
    plt.show()
    
    
def plot_corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(14,8)) # create sublots
    plt.title("Correlation Heatmap") # set the fig title
    mask = np.triu(df.corr()) # create upper triangle of an array

    sns.heatmap(df.corr(), annot = True, ax=ax, mask=mask, cmap='coolwarm', linewidths=.5) 
    
