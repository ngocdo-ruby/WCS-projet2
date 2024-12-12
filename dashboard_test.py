import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from st_aggrid import AgGrid
from st_keyup import st_keyup

# prend toute la largeur de la page
st.set_page_config(layout="wide")

df = pd.read_parquet('imdb.parquet')

# Analyse de données

# Sidebar
st.sidebar.title("Navigation")
pages = ["Dashboard", "Analyse des genres", "Analyse des acteurs", "Tendances temporelles"]
page = st.sidebar.radio("Choisissez une section :", pages)

# Fonction pour afficher des KPI cards
def display_kpis(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Films total", f"{len(df):,}")
    with col2:
        st.metric("Note moyenne", f"{df['averageRating'].mean():.2f}")
    with col3:
        st.metric("Votes totaux", f"{df['numVotes'].sum():,}")
    with col4:
        st.metric("Année la plus ancienne", int(df['year'].min()))

# Page 1 : Dashboard principal
if page == pages[0]:
    st.title("📊 Dashboard IMDB")
    display_kpis(df)

    st.divider()

    # Tabs pour organiser les graphiques
    tab1, tab2, tab3 = st.tabs(["Films par Genre", "Votes par Genre", "Nombre de Films par Année"])

    # Graphique : Nombre de films par genre
    with tab1:
        genre_counts = df['genres'].str.split(',').explode().value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Nombre de Films']
        fig_genre = px.bar(genre_counts, x='Nombre de Films', y='Genre', orientation='h',
                           title='Nombre de films par genre', color='Nombre de Films',
                           color_continuous_scale='greens')
        st.plotly_chart(fig_genre, use_container_width=True)

    # Graphique : Moyenne des votes par genre
    with tab2:
        df_genres = df[['genres', 'numVotes']].dropna()
        df_genres = df_genres.assign(genres=df_genres['genres'].str.split(',')).explode('genres')
        genre_avg_votes = df_genres.groupby('genres')['numVotes'].mean().reset_index()
        genre_avg_votes = genre_avg_votes.sort_values(by='numVotes', ascending=False)
        fig_avg_votes = px.bar(genre_avg_votes, x='genres', y='numVotes',
                               title='Moyenne des votes par genre', color='numVotes',
                               color_continuous_scale='oranges')
        st.plotly_chart(fig_avg_votes, use_container_width=True)

    # Graphique : Nombre de films par année
    with tab3:
        films_per_year = df['year'].value_counts().reset_index()
        films_per_year.columns = ['Année', 'Nombre de Films']
        films_per_year = films_per_year.sort_values(by='Année')
        fig_films_year = px.bar(films_per_year, x='Année', y='Nombre de Films',
                                title='Nombre de films par année', color='Nombre de Films',
                                color_continuous_scale='Blues')
        st.plotly_chart(fig_films_year, use_container_width=True)

# Page 2 : Analyse des genres
elif page == pages[1]:
    st.title("🌟 Analyse des Genres")
    genre_counts = df['genres'].str.split(',').explode().value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Nombre de Films']
    AgGrid(genre_counts)

# Page 3 : Analyse des acteurs
elif page == pages[2]:
    st.title("👨‍🎬 Analyse des Acteurs")
    df_actors = df[['actors_names', 'year']].dropna()
    df_actors = df_actors.assign(actors_names=df_actors['actors_names'].str.split(','))
    df_actors = df_actors.explode('actors_names').dropna()
    df_actors['actors_names'] = df_actors['actors_names'].str.strip()

    actor_counts = df_actors['actors_names'].value_counts().reset_index()
    actor_counts.columns = ['Acteur', 'Nombre de Films']
    actor_counts = actor_counts.head(50)

    st.subheader("Top 50 des acteurs les plus présents")
    fig_actors = px.bar(actor_counts, x='Nombre de Films', y='Acteur', orientation='h',
                        title='Acteurs les plus présents', color='Nombre de Films',
                        color_continuous_scale='Viridis')
    st.plotly_chart(fig_actors, use_container_width=True)

# Page 4 : Tendances temporelles
elif page == pages[3]:
    st.title("⏰ Tendances Temporelles")
    df_votes = df.groupby('year')['numVotes'].mean().reset_index()
    df_votes.columns = ['Année', 'Votes Moyens']
    fig_votes = px.line(df_votes, x='Année', y='Votes Moyens',
                        title='Evolution des votes moyens par année', markers=True)
    st.plotly_chart(fig_votes, use_container_width=True)

    st.subheader("Top 10 des films les plus populaires en 2023")
    df_2023 = df[df['year'] == 2023].sort_values(by='numVotes', ascending=False).head(10)
    st.table(df_2023[['title', 'numVotes', 'averageRating']].rename(columns={
        'title': 'Titre',
        'numVotes': 'Votes',
        'averageRating': 'Note Moyenne'
    }))

