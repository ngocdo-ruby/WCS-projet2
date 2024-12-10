import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st
import seaborn as sns   
import matplotlib.pyplot as plt
import plotly.express as px
from st_aggrid import AgGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA   
from sklearn.pipeline import Pipeline
from IPython.display import HTML
from st_keyup import st_keyup
import requests

# prend toute la largeur de la page
st.set_page_config(layout="wide")

df = pd.read_parquet('imdb.parquet')

# Initialisation des hyperparametres pour la modelisation
if 'hyperparametres' not in st.session_state:
    st.session_state.hyperparametres = {
        'n_voisins': 3,
        'type_metric': 'euclidean'  
    }
    
st.sidebar.title("Sommaire")

pages = ["projet", "Exploration des données", "Analyse de données", "Modélisation", "Application de recommandations"]

# Initialise st.session_state.page si ce n'est pas déjà fait
if 'page' not in st.session_state:
    st.session_state.page = pages[0]  # Défini la page par défaut à "projet"

page = st.sidebar.radio("Pages", pages, index=pages.index(st.session_state.page), label_visibility='hidden')

# Met à jour la session_state pour la page sélectionnée
st.session_state.page = page

if page == pages[0]:
    st.header("Objectif")

    st.write("Réaliser une analyse approfondie de la base de données pour identifier des tendances et caractéristiques spécifiques.")
    st.write("Cette analyse devrait inclure : l’identification des acteurs les plus présents et les périodes associées, l’évolution de la durée moyenne des films au fil des années, la comparaison entre les acteurs présents au cinéma et dans les séries, l’âge moyen des acteurs, ainsi que les films les mieux notés et les caractéristiques qu’ils partagent.")

    st.write("Sur la base des informations récoltées, vous pourrez affiner votre programmation en vous spécialisant par exemple sur les films des années 90 ou les genres d’action et d’aventure, afin de mieux répondre aux attentes du public identifié lors de l’étude de marché")

    st.header("Besoin client")

    st.write("Obtenir quelques statistiques sur les films (type, durée), acteurs (nombre de films, type de films) et d’autres.")
# Exploration des données
elif page == pages[1]:
    st.header("Exploration des données")

    st.write("Dataframe :")
    st.dataframe(df, height=400)
    
    st.write("Dimensions du dataframe :")
    st.write(df.shape)
    
    st.write("Describe :")
    st.write(df.describe().T)
     
    if st.checkbox("Afficher les valeurs manquantes") :
        st.dataframe(df.isna().sum())

    if st.checkbox("Afficher les doublons") :
        st.write(df.duplicated().sum())

# Analyse de données
elif page == pages[2]:
    st.header("Analyse de données")

    df[['year', 'duration', 'averageRating', 'numVotes']] = df[['year', 'duration', 'averageRating', 'numVotes']].astype("float")

    # Pairplot des variables numériques
    fig_scatter_matrix = px.scatter_matrix(df,
                                        dimensions=['year', 'duration', 'averageRating', 'numVotes'],
                                        title="Matrice de dispersion des variables numériques",
                                        color='averageRating',
                                        labels={'year': 'Année', 'duration': 'Durée', 'averageRating': 'Note moyenne', 'numVotes': 'Nombre de votes'},
                                        color_continuous_scale='Viridis')
    fig_scatter_matrix.update_layout(
        title_font_size=26,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        height=800
    )
    fig_scatter_matrix.update_traces(diagonal_visible=False,  showupperhalf=False)
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)



    # calcul nombre de films par genre
    genre_counts = df['genres'].str.split(',').explode().value_counts()
    genre_counts_df = genre_counts.reset_index()
    genre_counts_df.columns = ['Genre', 'Count']

    # barplot du nombre de films par genre
    genre_counts_df = genre_counts_df.sort_values(by='Count', ascending=True)
    fig_genre_count = px.bar(genre_counts_df,
                            x='Count',
                            y='Genre',
                            title='Nombre de films par genre',
                            color='Count',
                            color_continuous_scale='greens',
                            labels={'Genre': 'Genre', 'Count': 'Nombre de films'},
                            text='Count')
    fig_genre_count.update_layout(
        title_font_size=26,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        height=800
    )
    st.plotly_chart(fig_genre_count, use_container_width=True)


    # calcul moyenne du nombre de votes par genre
    df_genres = df[['genres', 'numVotes']].dropna()
    df_genres = df_genres.assign(genres=df_genres['genres'].str.split(',')).explode('genres')
    genre_avg_votes = df_genres.groupby('genres')['numVotes'].mean().reset_index()

    genre_counts_df = genre_counts_df.merge(genre_avg_votes, left_on='Genre', right_on='genres', how='left')
    genre_counts_df.drop(columns='genres', inplace=True)
    genre_counts_df = genre_counts_df.sort_values(by='numVotes', ascending=False)

    # barplot moyenne des votes par genre
    fig_avg_votes_per_genre = px.bar(genre_counts_df,
                                    x='Genre',
                                    y='numVotes',
                                    title='Moyenne des votes par genre',
                                    color='numVotes',
                                    color_continuous_scale='oranges',
                                    labels={'Genre': 'Genre', 'numVotes': 'Moyenne des votes'}
                                    )
    fig_avg_votes_per_genre.update_layout(
        xaxis_tickangle=-45,
        title_font_size=26,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        height=800
    )
    st.plotly_chart(fig_avg_votes_per_genre, use_container_width=True)

    # Calcul nombre de films par année
    film_count_per_year = df['year'].value_counts().sort_index()
    df_nb_film_per_year = film_count_per_year.reset_index()
    df_nb_film_per_year.columns = ['year', 'nombre_de_films']

    #barplot du nombre de films par année
    fig_nb_film_per_year = px.bar(df_nb_film_per_year,
              x='year',
              y='nombre_de_films',
              title='Nombre de films par année',
              color='nombre_de_films',
              color_continuous_scale='Blues',
              labels={'year': 'Année', 'nombre_de_films': 'Nombre de films'})

    fig_nb_film_per_year.update_layout(
        title_font_size=26,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        height=800
    )
    st.plotly_chart(fig_nb_film_per_year, use_container_width=True)

    # liste d'acteurs par film et apparitions
    df_actors = df[['actors_names', 'year']].dropna()
    df_actors = df_actors.assign(actors_names=df_actors['actors_names'].str.split(','))
    df_actors = df_actors.explode('actors_names')
    df_actors['actors_names'] = df_actors['actors_names'].str.strip()

    actor_counts = df_actors['actors_names'].value_counts()
    actor_counts_df = actor_counts.reset_index()
    actor_counts_df.columns = ['Acteur', 'Apparitions']

    # Identifie les années d'activité des acteurs
    actor_years = df_actors.groupby('actors_names')['year'].unique().reset_index()
    actor_activity_df = pd.merge(actor_counts_df, actor_years, left_on='Acteur', right_on='actors_names')
    actor_activity_df = actor_activity_df.sort_values(by='Apparitions', ascending=False).head(50)
    actor_activity_df = actor_activity_df[['Acteur', 'Apparitions', 'year']]
    actor_activity_df.rename(columns={'year': 'Années'}, inplace=True)

    st.divider()

    # Tableau des acteurs les plus présents
    st.header("Top 50 des acteurs les plus présents et leurs périodes d'activité")
    AgGrid(actor_activity_df)

    ############# PAS DE WIDTH 100% POSSIBLE #############
    # st.dataframe(actor_activity_df)
    ######################################################


    #  barplot des acteurs les plus présents
    fig_actor_activity = px.bar(actor_activity_df,
                                x='Acteur',
                                y='Apparitions',
                                color='Apparitions',
                                title='Acteurs les plus présents dans les films',
                                color_continuous_scale='Viridis',
                                labels={'Acteur': 'Acteur', 'Apparitions': 'Nombre de films'})
    fig_actor_activity.update_layout(
        xaxis_tickangle=-45,
        title_font_size=26,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        height=800
    )
    st.plotly_chart(fig_actor_activity, use_container_width=True)
    
    #####################CAMILLE################################

    # Rating VS year chart:
    df_nbr_votes_by_year = df[["year", "numVotes"]]
    df_nbr_votes_by_year = df_nbr_votes_by_year.groupby("year", as_index=False)[["numVotes"]].mean()
    # round the mean values
    df_nbr_votes_by_year["numVotes_moyenne"] = df_nbr_votes_by_year["numVotes"].apply(lambda x : round(x, 2))
    # generate a bar chart
    fig_rating_vs_years = px.bar(df_nbr_votes_by_year, x="year", y="numVotes_moyenne", color_continuous_scale='Blues', color="numVotes_moyenne")
    st.write("### Graph évolution des votes par années")
    st.plotly_chart(fig_rating_vs_years)

    # 10 most populaire movies in 2023
    st.write("### Les films les plus populaires en 2023")

     # custom data frame
    df_data_2023 = df[df["year"] == 2023.0]
    df_rating = df_data_2023[["title", "numVotes", "averageRating"]]
    sorted_df = df_rating.sort_values(by=["numVotes"], ascending=False)
    # take the top 10 movies
    resultat = sorted_df[0:10]
    # rename columns and return dataframe
    df_populare_movies = resultat.rename(columns={"title": "Titre", "numVotes": "Votes", "averageRating": "Note moyenne"})
    st.dataframe(df_populare_movies, hide_index=True)

# Modélisation  
elif page == pages[3]:
    st.header("Modélisation")    

    # Standardise les données numériques
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['number']).columns
    if 'df_scaled' not in st.session_state:
        df_scaled = scaler.fit_transform(df[num_cols])
        st.session_state.df_scaled = df_scaled
    else:
        df_scaled = st.session_state.df_scaled

    st.write('Colonnes numériques traités par StandardScaler :')
    st.write(list(num_cols))
    
    st.write('Données Scaled ( head() ) :')
    st.write(pd.DataFrame(df_scaled, columns=num_cols).head())
    
    # Créer le modèle NearestNeighbors
    # Choix des hyperparametres
    st.write('Choix des hyperparametres pour le modèle NearestNeighbors :')
    st.session_state.hyperparametres['n_voisins'] = st.number_input('Nombre de voisins', min_value=1, max_value=5, value=st.session_state.hyperparametres['n_voisins'])

    st.session_state.hyperparametres['type_metric'] = st.segmented_control(
        "Metric", ["euclidean", "manhattan", "minkowski", "chebyshev", "cosine"],
        selection_mode="single",
        default=st.session_state.hyperparametres['type_metric'] 
    )
    nn = NearestNeighbors(n_neighbors=st.session_state.hyperparametres['n_voisins'] + 1, metric=st.session_state.hyperparametres['type_metric'])
    nn.fit(df_scaled)
    
    # Sauvegarde le modèle dans session_state pour l'utiliser dans d'autres pages
    st.session_state.nn_model = nn
    st.session_state.df_scaled_session = df_scaled

# Application de recommandations  
elif page == pages[4]:
    st.header("Application de recommandations")
    
    # Accéde au modèle NearestNeighbors depuis session_state
    if 'nn_model' in st.session_state:
        nn = st.session_state.nn_model
    else:
        # force la page a modelisation pour initier le model
        st.session_state.page = pages[3]  
        
    if 'df_scaled_session' in st.session_state:
        df_scaled = st.session_state.df_scaled
        
    # Fonction pour rechercher des films
    def search_movies(query, df):
        if not query:
            return pd.DataFrame(columns=df.columns)
        return df[df['title'].str.contains(query, case=False, na=False)]
    
    def find_neighbors_title(title, df):        
        # films correspondant au titre donné
        filtered_df = df[df['title'].str.lower() == title.lower()]

        # Vérifie si le film existe dans le DataFrame
        if filtered_df.empty:
            st.write(f"Film '{title}' non trouvé.")
            return None  
        
        # Index du 1er film avec le title donné
        index_imdb = filtered_df.index[0]
        
        # Voisins du film
        distances, indices = nn.kneighbors(df_scaled[index_imdb].reshape(1, -1))
        
        # Exclu le film lui-même des voisins
        filtered_indices = [i for i in indices[0] if i != index_imdb]
        
        # Retourner les voisins et leurs informations
        neighbors_df = df.iloc[filtered_indices]
        
        for index, row in neighbors_df.iterrows():
            with st.container():
                    st.markdown(f"""
                        <div class="movie-container">
                            <div style="display: flex; align-items: center;">
                                <img src="{row['poster_path']}" alt="Poster" style="width: 120px; margin-right: 20px;" class="poster">
                                <div>
                                    <div class="titre">{row['title']} ({row['year']})</div>
                                    <div class="genre">Genres: {row['genres']}</div>
                                    <div class="rating">⭐ {row['averageRating']}/10</div>
                                    <div class="description">{row['overview']}</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            st.write("") 
            
        return neighbors_df
      
    # Saisie du film avec recherche systematique a chauqe lettre tappée
    film = st_keyup("Saisissez un film", debounce=500, key="2")

    # Recherche les films correspondant au titre saisi
    results = search_movies(film, df)
    
    # Affiche les résultats
    if results.empty:
        st.write("Aucun film trouvé.")
    else:
        st.write(f"{len(results)} film(s) trouvé(s) :")     
        # Liste des films correspondants       
        selected_movie = st.pills("films", results['title'], selection_mode="single", default=None,  label_visibility='hidden')
        if selected_movie:
            # Trouver le film sélectionné dans le dataframe 'results'
            selected_row = results[results['title'].str.lower() == selected_movie.lower()].iloc[0]

            "---"
            st.header(f"Film sélectionné : {selected_row['title']}")
            # Affichage des informations détaillées du film
            with st.container():
                st.markdown(f"""
                    <div class="movie-container">
                        <div style="display: flex; align-items: center;">
                            <img src="{selected_row['poster_path']}" alt="Poster" style="width: 120px; margin-right: 20px;" class="poster">
                            <div>
                                <div class="titre">{selected_row['title']} ({selected_row['year']})</div>
                                <div class="genre">Genres: {selected_row['genres']}</div>
                                <div class="rating">⭐ {selected_row['averageRating']}/10</div>
                                <div class="description">{selected_row['overview']}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            "---"
            # Affichage de la liste des voisins
            st.header(f"Les {st.session_state.hyperparametres['n_voisins'] } voisins :")
            neighbors = find_neighbors_title(selected_row['title'], df)
            st.write("")
            st.dataframe(neighbors)  # Affichage du dataframe correspondant
