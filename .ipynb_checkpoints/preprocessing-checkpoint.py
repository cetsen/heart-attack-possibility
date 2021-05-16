import pandas as pd
import numpy as np


def clean(df):
    print("Number of duplicates in dataframe:", df.duplicated().sum(), "-- cleaning duplicates")
    # Keep only the first of the duplicates
    df = df[~df.duplicated(keep='first')]

    print("Number of rows with ca value large than 3:", len(df[df.ca > 3]), "-- removing erronous rows")
    # Remove erronous rows (with ca value > 3)
    df = df[df.ca <= 3]

    print("Number of samples after cleaning:", len(df))
    
    return df