import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st
from st_clickable_images import clickable_images
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
import actors

# prend toute la largeur de la page
st.set_page_config(layout="wide")

df = pd.read_parquet('imdb.parquet')

API_KEY = '60c5ca9b75de2d2e768380e9a5bfd88c'
MAIN_URL = "https://api.themoviedb.org/3"

# Initialisation des hyperparametres pour la modelisation
if 'hyperparametres' not in st.session_state:
    st.session_state.hyperparametres = {
        'n_voisins': 3,
        'type_metric': 'euclidean'
    }

st.sidebar.title("Sommaire")

pages = ["Projet", "Exploration des données", "Analyse de données", "Dashboard", "Modélisation", "Application de recommandations"]

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

# Dashboard
elif page == pages[3]:
    st.header("Dashboard")

    # Fonction pour afficher des KPI cards
    def display_kpi(df):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Films total", f"{len(df):,}")
        with col2:
            st.metric("Note moyenne", f"{df['averageRating'].mean():.2f}")
        with col3:
            st.metric("Votes totaux", f"{int(df['numVotes'].sum()):,}")
        with col4:
            st.metric("Année la plus ancienne", int(df['year'].min()))
        with col5:
            st.metric("Année la plus récente", int(df['year'].max()))


    st.title("📊 Dashboard")
    display_kpi(df)  # Afficher les KPI

    st.divider()

    # Tabs pour organiser les graphiques
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Films par genre", "Votes par genre", "Films par année", "Analyse des acteurs", "Tendances temporelles"])

    # Nombre de films par genre
    with tab1:
        genre_counts = df['genres'].str.split(',').explode().value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Nombre de films']
        fig_genre = px.bar(genre_counts, x='Nombre de films', y='Genre', orientation='h',
                            title='Nombre de films par genre', color='Nombre de films',
                            color_continuous_scale='oranges')
        st.plotly_chart(fig_genre, use_container_width=True)

    # Moyenne de nombre de votes par genre
    with tab2:
        df_genres = df[['genres', 'numVotes']].dropna()
        df_genres = df_genres.assign(genres=df_genres['genres'].str.split(',')).explode('genres')
        genre_avg_votes = df_genres.groupby('genres')['numVotes'].mean().reset_index()
        genre_avg_votes = genre_avg_votes.sort_values(by='numVotes', ascending=False)
        fig_avg_votes = px.bar(genre_avg_votes, x='genres', y='numVotes',
                                title='Moyenne de nombre de votes par genre', color='numVotes',
                                color_continuous_scale='oranges')
        st.plotly_chart(fig_avg_votes, use_container_width=True)

    # Nombre de films par année
    with tab3:
        films_per_year = df['year'].value_counts().reset_index()
        films_per_year.columns = ['Année', 'Nombre de films']
        films_per_year = films_per_year.sort_values(by='Année')
        fig_films_year = px.bar(films_per_year, x='Année', y='Nombre de films',
                                title='Nombre de films par année', color='Nombre de films',
                                color_continuous_scale='oranges')
        st.plotly_chart(fig_films_year, use_container_width=True)

    # Analyse des acteurs
    with tab4:
        st.title("👨‍🎬 Analyse des acteurs")
        df_actors = df[['actors_names', 'year']].dropna()
        df_actors = df_actors.assign(actors_names=df_actors['actors_names'].str.split(','))
        df_actors = df_actors.explode('actors_names').dropna()
        df_actors['actors_names'] = df_actors['actors_names'].str.strip()

        actor_counts = df_actors['actors_names'].value_counts().reset_index()
        actor_counts.columns = ['Acteur', 'Nombre de films']
        actor_counts = actor_counts.head(50)

        st.subheader("Top 50 des acteurs les plus présents")
        fig_actors = px.bar(actor_counts, x='Nombre de films', y='Acteur', orientation='h',
                            color='Nombre de films',
                            color_continuous_scale='oranges')
        st.plotly_chart(fig_actors, use_container_width=True)

    # Tendances temporelles
    with tab5:
        st.title("⏰ Tendances Temporelles")

        st.subheader("Evolution des votes par année")
        df_votes = df.groupby('year')['numVotes'].mean().reset_index()
        df_votes.columns = ['Année', 'Votes moyens']
        fig_votes = px.line(df_votes, x='Année', y='Votes moyens', markers=True)
        st.plotly_chart(fig_votes, use_container_width=True)

        st.subheader("Top 10 des films les plus populaires en 2023")
        df_2023 = df[df['year'] == 2023].sort_values(by='numVotes', ascending=False).head(10)
        df_2023['numVotes'] = df_2023['numVotes'].apply(lambda x: f"{int(x):,}")
        df_2023['averageRating'] = df_2023['averageRating'].apply(lambda x: f"{x:.1f}")
        st.table(df_2023[['title', 'numVotes', 'averageRating']].rename(columns={
            'title': 'Titre',
            'numVotes': 'Votes',
            'averageRating': 'Note moyenne'
        }))


# Modélisation
elif page == pages[4]:
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
elif page == pages[5]:
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

    def get_tmdb_data(imdb_id):
        # L'URL de base pour l'API TMDb
        url = f'https://api.themoviedb.org/3/find/{imdb_id}?language=fr-FR'
        params = {
            'api_key': API_KEY,
            'external_source': 'imdb_id'  # Indique que nous recherchons avec l'IMDb ID
        }
        # Envoi de la requête
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'movie_results' in data and len(data['movie_results']) > 0:
                movie = data['movie_results'][0]  # On prend le premier résultat (si disponible)
                return movie
        return None

    def get_tmdb_video(tmdb_id):
        url = f'https://api.themoviedb.org/3/movie/{tmdb_id}/videos?language=fr-FR'
        params = {
            'api_key': API_KEY
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                video = data['results'][0]
                return video
        return None

    def get_tmdb_actors(tmdb_id):
        url = f'https://api.themoviedb.org/3/movie/{tmdb_id}/credits?language=fr-FR'
        params = {
            'api_key': API_KEY
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            #recupere les 5 premiers acteurs
            if 'cast' in data and len(data['cast']) > 0:
                actors = data['cast'][:5]
                return actors
        return None

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

            # Récupération des données TMDb pour le film sélectionné
            tmdb = get_tmdb_data(selected_row['imdb_id'])

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
                                <div class="description">{tmdb.get('overview', None)}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.write("")

            #affichage des acteurs
            tmdb_actors = get_tmdb_actors(tmdb['id'])
            st.header("Acteurs principaux")
            # Créer un container pour le bandeau
            with st.container():
                cols = st.columns(len(tmdb_actors))
                i = 0# une colonne par acteur

                for col, actor in zip(cols, tmdb_actors):
                    with col:
                        st.image(f"https://media.themoviedb.org/t/p/w138_and_h175_face{actor['profile_path']}", caption=actor['name'], use_container_width='auto')
                        if st.button(actor["name"]):
                            id_actor = actor['id']
                            actor_credits = actors.get_movies(id_actor)
                            with st.container():
                               st.html(
                                    f"<p><b>Date de naissance</b>: {actors.get_birthdate(id_actor)}</p>"
                                    f"<p><b/>Biographie:</b> {actors.get_biography(id_actor)}</p>"
                                    )
                               st.divider()
                               st.html(f"<h3>Filmographie:</h3>")
                            #    clicked = clickable_images(actors.list_movie_posters(id_actor))
                            #    if clicked > -1:
                            #         st.markdown(f"Image #{clicked} clicked" )
                               for title, year in actor_credits.items():
                                   st.html(f"<li>{title}: {year}</li>")
            st.write("")

            # video si elle existe
            vid = get_tmdb_video(tmdb['id'])
            if vid and vid['site'] == 'YouTube':
                col1,col2=st.columns([2,2])
                with col1:
                    st.video(f"https://www.youtube.com/watch?v={vid['key']}")

            "---"
            # Affichage de la liste des voisins
            st.header(f"Les {st.session_state.hyperparametres['n_voisins'] } voisins :")
            neighbors = find_neighbors_title(selected_row['title'], df)
            st.write("")
            st.dataframe(neighbors)  # Affichage du dataframe correspondant
