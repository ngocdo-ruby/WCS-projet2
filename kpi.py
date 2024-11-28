import pandas as pd 
import numpy as np 
from collections import Counter
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import r2_score

df = pd.read_parquet('imdb_28112024.parquet')

st.sidebar.title("Sommaire")

pages = ["projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("", pages)

if page == pages[0] : 
    
    st.write("### Objectif")
    
    st.write("Réaliser une analyse approfondie de la base de données pour identifier des tendances et caractéristiques spécifiques.")
    st.write("Cette analyse devrait inclure : l’identification des acteurs les plus présents et les périodes associées, l’évolution de la durée moyenne des films au fil des années, la comparaison entre les acteurs présents au cinéma et dans les séries, l’âge moyen des acteurs, ainsi que les films les mieux notés et les caractéristiques qu’ils partagent.")
    
    st.write("Sur la base des informations récoltées, vous pourrez affiner votre programmation en vous spécialisant par exemple sur les films des années 90 ou les genres d’action et d’aventure, afin de mieux répondre aux attentes du public identifié lors de l’étude de marché")
    
    st.write("### Besoin client")
    
    st.write("Obtenir quelques statistiques sur les films (type, durée), acteurs (nombre de films, type de films) et d’autres.")
    
elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
        
elif page == pages[2]:
    st.write("### Analyse de données")

    df[['startYear', 'duration', 'averageRating', 'numVotes']] = df[['startYear', 'duration', 'averageRating', 'numVotes']].astype("float")

    fig_pairplot = sns.pairplot(data=df[['startYear', 'duration', 'averageRating', 'numVotes']], corner=True)
    st.pyplot(fig_pairplot)

    fig_scatter_matrix = px.scatter_matrix(df,
                                        dimensions=['startYear', 'duration', 'averageRating', 'numVotes'],
                                        title="Matrice de dispersion des variables numériques",
                                        color='averageRating', 
                                        labels={'startYear': 'Année', 'duration': 'Durée', 'averageRating': 'Note moyenne', 'numVotes': 'Nombre de votes'},
                                        color_continuous_scale='Viridis')  
    st.plotly_chart(fig_scatter_matrix)

    # nombre de films par genre
    genre_counts = df['genres'].str.split(',').explode().value_counts()
    genre_counts_df = genre_counts.reset_index()
    genre_counts_df.columns = ['Genre', 'Count']

    fig_genre_count = px.bar(genre_counts_df,
                            x='Genre',
                            y='Count',
                            title='Nombre de films par genre',
                            color='Count', 
                            color_continuous_scale='Viridis',
                            labels={'Genre': 'Genre', 'Count': 'Nombre de films'},
                            text='Count')  
    fig_genre_count.update_layout(xaxis_tickangle=-45, width=1000, height=600)
    st.plotly_chart(fig_genre_count)

    # moyenne du nombre de votes par genre et fusionner avec les comptages
    df_genres = df[['genres', 'numVotes']].dropna()
    df_genres = df_genres.assign(genres=df_genres['genres'].str.split(',')).explode('genres')
    genre_avg_votes = df_genres.groupby('genres')['numVotes'].mean().reset_index()

    genre_counts_df = genre_counts_df.merge(genre_avg_votes, left_on='Genre', right_on='genres', how='left')
    genre_counts_df.drop(columns='genres', inplace=True)
    genre_counts_df = genre_counts_df.sort_values(by='numVotes', ascending=False)

    # moyenne des votes par genre
    fig_avg_votes_per_genre = px.bar(genre_counts_df,
                                    x='Genre',
                                    y='numVotes',
                                    title='Moyenne des votes par genre',
                                    color='numVotes',  # Couleur des barres selon la moyenne des votes
                                    color_continuous_scale='Viridis',
                                    labels={'Genre': 'Genre', 'numVotes': 'Moyenne des votes'},
                                    text='numVotes')  # Afficher la moyenne des votes sur chaque barre
    fig_avg_votes_per_genre.update_layout(xaxis_tickangle=-45, width=1000, height=600)
    st.plotly_chart(fig_avg_votes_per_genre)

    # liste d'acteurs par film et apparitions
    df_actors = df[['actors_names', 'startYear']].dropna()
    df_actors = df_actors.assign(actors_names=df_actors['actors_names'].str.split(','))
    df_actors = df_actors.explode('actors_names')
    df_actors['actors_names'] = df_actors['actors_names'].str.strip()

    actor_counts = df_actors['actors_names'].value_counts()
    actor_counts_df = actor_counts.reset_index()
    actor_counts_df.columns = ['Acteur', 'Apparitions']

    # Identifier les années d'activité des acteurs
    actor_years = df_actors.groupby('actors_names')['startYear'].unique().reset_index()
    actor_activity_df = pd.merge(actor_counts_df, actor_years, left_on='Acteur', right_on='actors_names')
    actor_activity_df = actor_activity_df.sort_values(by='Apparitions', ascending=False).head(100)
    actor_activity_df = actor_activity_df[['Acteur', 'Apparitions', 'startYear']]

    st.write("### Acteurs les plus présents et leurs périodes d'activité")
    st.write(actor_activity_df)

    #  barres des acteurs les plus présents
    fig_actor_activity = px.bar(actor_activity_df,
                                x='Acteur',
                                y='Apparitions',
                                color='Apparitions',
                                title='Acteurs les plus présents dans les films',
                                color_continuous_scale='Viridis',
                                labels={'Acteur': 'Acteur', 'Apparitions': 'Nombre de films'})
    fig_actor_activity.update_layout(xaxis_tickangle=-45, width=1000, height=600)
    st.plotly_chart(fig_actor_activity)


    
    
    #################RUBY################################
    # Compter les genres  
    genre_counts = Counter()  
    # Boucle pour compter les genres
    for genre_list in df['genres']:  
        genres = genre_list.split(',')  # Séparer les genres  
        genre_counts.update(genres)  # Mettre à jour le compteur  
    # Convertir le compteur en DataFrame  
    df_genre_counts = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Nombre'])  
    df_genre_counts = df_genre_counts.sort_values(by='Nombre', ascending=True)
    
    # Créer l'histogramme  
    fig2 = plt.figure(figsize=(20, 6))  
    sns.barplot(x='Nombre', y='Genre', data=df_genre_counts, color='green')  
    plt.title('Les genres de films les plus fréquents', fontsize=16)  
    plt.xlabel('Nombre de films', fontsize=14)  
    plt.ylabel('Genre', fontsize=14)    
    st.pyplot(fig2)
    
    
    
    fig22 = px.bar(df_genre_counts, 
                   x='Nombre', 
                   y='Genre', 
                   title='Les genres de films les plus fréquents',
                   color='Nombre',  
                   color_continuous_scale='greens',  
                   labels={'Nombre': 'Nombre de films', 'Genre': 'Genre'}
                   )  
    st.plotly_chart(fig22)
    
    
    
    
    # Compter le nombre de films par année
    film_count_per_year = df['startYear'].value_counts().sort_index()
    # Conversion en DataFrame
    df_nb_film_per_year = film_count_per_year.reset_index()
    # Renommer les colonnes
    df_nb_film_per_year.columns = ['startYear', 'nombre_de_films']
    # Créer l'histogramme
    fig3 = plt.figure(figsize=(30, 10))
    sns.barplot(x='startYear', y='nombre_de_films', data=df_nb_film_per_year, color='blue')
    plt.title('Nombre de films par année', fontsize=26)
    plt.xlabel('Année', fontsize=18)
    plt.ylabel('Nombre de films', fontsize=18)
    st.pyplot(fig3)
    
    
    
    fig32 = px.bar(df_nb_film_per_year, 
              x='startYear', 
              y='nombre_de_films', 
              title='Nombre de films par année',
              color='nombre_de_films',  
              color_continuous_scale='Blues', 
              labels={'startYear': 'Année', 'nombre_de_films': 'Nombre de films'}) 

    fig32.update_layout(
        width=900,  
        height=300, 
        title_font_size=26,  
        xaxis_title_font_size=18, 
        yaxis_title_font_size=18, 
    )    
    st.plotly_chart(fig32)
    #####################################################

   
elif page == pages[3]:
    st.write("### Modélisation")