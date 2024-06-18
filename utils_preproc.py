import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns
import matplotlib.lines as mlines

""" PRE-PROCESSING FUNCTIONS """

def power_definer(df):
    df['party_in_power'] = 0
    df.loc[df['party_status'] == 'Coalition', 'party_in_power'] = 1
    df.drop(columns=['party_status'], inplace=True)

def age_getter(df):
    df['date'] = pd.to_datetime(df['date'])
    df['speaker_age'] = df['date'].dt.year - df['speaker_birth']
    df.drop(columns=['speaker_birth'], inplace=True)

def brickwall_limiter(df, column, max_tokens):
    df['speech_text_preprocessed_len'] = df[column].apply(lambda x: len(x))
    filtered_df = df.loc[df['speech_text_preprocessed_len'] > max_tokens]
    return filtered_df

def split_token_lists_in_df(df, token_column, max_tokens, keep_columns=None):
    """
    Splits lists of tokens in a specified column into even parts, adds these parts as new rows,
    and retains the values of all other columns.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - token_column (str): The name of the column containing the lists of tokens to split.
    - max_tokens (int): The maximum number of tokens per sublist.
    - keep_columns (list): List of other column names to retain in each new row. If None, keeps all columns.

    Returns:
    - pd.DataFrame: A new DataFrame with the original rows having long lists removed and new rows added for each sublist of tokens.
    """
    
    # Function to split the list of tokens into even parts
    def split_tokens(tokens, n):
        return [tokens[i:i + n] for i in range(0, len(tokens), n)]
    
    # Initialize a list to hold the new rows
    new_rows = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Split the token list if it's longer than max_tokens
        if len(row[token_column]) > max_tokens:
            parts = split_tokens(row[token_column], max_tokens)
            for part in parts:
                new_row = row.copy()
                new_row[token_column] = part
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    
    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows).reset_index(drop=True)
    
    # If keep_columns is not None, filter the DataFrame to keep only the specified columns
    if keep_columns is not None:
        new_df = new_df[keep_columns + [token_column]]
    
    return new_df

def limit_splitter(df, max_tokens):
    df = split_token_lists_in_df(df, 'speech_text_preprocessed_tokenized', max_tokens)
    df['speech_text_preprocessed_len'] = df['speech_text_preprocessed_tokenized'].apply(lambda x: len(x))
    return df

""" PLOT FUNCTIONS """

def cool_party_plot(pred_df, color1='red', color2='skyblue', label1='PSOE', label2='PP'):
    # Two subplots
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1})

    # Plot density plots for each class on the first subplot (axes[0])
    sns.kdeplot(data=pred_df[pred_df['true'] == 0]['pred'], ax=axes[0], color=color1, fill=True, bw_adjust=0.5)
    sns.kdeplot(data=pred_df[pred_df['true'] == 1]['pred'], ax=axes[0], color=color2, fill=True, bw_adjust=0.5)
    axes[0].set_ylabel('Density')
    axes[0].set_ylim(bottom=0)

    # Custom legend for the bottom plot
    legend_class_0 = mlines.Line2D([], [], color=color1, marker='_', linestyle='None', markersize=10, label=label1)
    legend_class_1 = mlines.Line2D([], [], color=color2, marker='_', linestyle='None', markersize=10, label=label2)
    axes[0].legend(handles=[legend_class_0, legend_class_1], title='True Party')

    # Plot vertical lines for each observation on the second subplot (axes[1])
    for _, row in pred_df.iterrows():
        axes[1].vlines(row['pred'], ymin=0, ymax=1, color=color2 if row['true'] == 1 else color1, alpha=0.7)


    # Common settings for the whole figure
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_yticks([])
    plt.xlim([0.0, 1.0])
    plt.suptitle('Predicted Probabilities and True Classes', fontsize=15)

    plt.show()