# %%
import pandas as pd

def populareMoviesDf(df):
    # custom data frame
    df_data_2023 = df[df["startYear"] == "2023"]
    df_data_2023
    df_rating = df_data_2023[["title", "numVotes", "averageRating"]]
    sorted_df = df_rating.sort_values(by=["numVotes"], ascending=False)

    # take the top 10 movies
    resultat = sorted_df[0:10]

    # rename columns and return dataframe
    resultat = resultat.rename(columns={"title": "Titre", "numVotes": "Votes", "averageRating": "Note moyenne"})
    resultat


