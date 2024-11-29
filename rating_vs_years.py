import pandas as pd
import plotly.express as px

def charRatingVsYears(df):
    # group by year
    df_nbr_votes_by_year = df[["startYear", "numVotes"]]
    df_nbr_votes_by_year = df_nbr_votes_by_year.groupby("startYear", as_index=False)[["numVotes"]].mean()

    # round the mean values
    df_nbr_votes_by_year["numVotes_moyenne"] = df_nbr_votes_by_year["numVotes"].apply(lambda x : round(x, 2))
    df_nbr_votes_by_year

    # generate a bar chart
    px.bar(df_nbr_votes_by_year, x="startYear", y="numVotes_moyenne")


