import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from st_aggrid import AgGrid
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from st_keyup import st_keyup
import requests

# prend toute la largeur de la page
st.set_page_config(layout="wide")

df = pd.read_parquet("imdb.parquet")

API_KEY = "60c5ca9b75de2d2e768380e9a5bfd88c"

st.sidebar.title("Sommaire")

pages = [
    "Dashboard",    
    "Application de recommandations",
]

# Initialise st.session_state.page si ce n'est pas déjà fait
if "page" not in st.session_state:
    st.session_state.page = pages[0]  # Défini la page par défaut à "projet"

page = st.sidebar.radio(
    "Pages", pages, index=pages.index(st.session_state.page), label_visibility="hidden"
)

# Met à jour la session_state pour la page sélectionnée
st.session_state.page = page

if page == pages[0]:
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
    
elif page == pages[1]:
    st.header("Application de recommandations")

    # Fonction pour rechercher des films
    def search_movies(query, df):
        if not query:
            return pd.DataFrame(columns=df.columns)
        return df[df["title"].str.contains(query, case=False, na=False)]

     # Fonction pour recuperer les donnees
    def get_tmdb_data(imdb_id):
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?language=fr-FR"
        params = {
            "api_key": API_KEY,
            "external_source": "imdb_id",  # nous recherchons avec l'IMDb ID
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "movie_results" in data and len(data["movie_results"]) > 0:
                movie = data["movie_results"][
                    0
                ]  # On prend le premier résultat (si disponible)

                # Ajouter les genres en français dans l'objet movie
                genres = movie.get("genre_ids", [])
                genres_dict = (
                    get_tmdb_genres()
                )  # Récupère la liste des genres en français
                genre_names = [
                    genres_dict.get(genre_id, "Inconnu") for genre_id in genres
                ]

                # Ajouter les genres traduits au film
                movie["genre_names"] = genre_names
                movie[
                    "genre_ids"
                ] = genres  # Assurez-vous que genre_ids est aussi inclus

                return movie
        return None

    # Fonction pour recuperer les genres en fr
    def get_tmdb_genres():
        url = "https://api.themoviedb.org/3/genre/movie/list?language=fr-FR"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "genres" in data:
                # Créer un dictionnaire pour mapper les ID de genres à leurs noms en français
                genres_dict = {genre["id"]: genre["name"] for genre in data["genres"]}
                return genres_dict
        return None

    # Fonction pour recuperer la video 
    def get_tmdb_video(tmdb_id):
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos?language=fr-FR"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                video = data["results"][0]
                return video
        return None

    # Fonction pour recuperer les infos des acteurs
    def get_tmdb_actors(tmdb_id):
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits?language=fr-FR"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            # recupere les 5 premiers acteurs
            if "cast" in data and len(data["cast"]) > 0:
                actors = data["cast"][:5]
                return actors
        return None

    # Fonction pour recuperer la tagline du film
    def get_tmdb_tagline(tmdb_id):
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=fr-FR"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "tagline" in data:
                return data["tagline"]
        return None

    # Fonction pour afficher les details du film selectionné
    def display_movie_info(selected_row, tmdb):
        # Affichage des informations du film
        with st.container():
            st.markdown(
                f"""
                <div class="movie-container">
                    <div style="display: flex; align-items: center;">
                        <img src="https://image.tmdb.org/t/p/w500/{tmdb.get('poster_path', None)}" alt="Poster" style="width: 300px; margin-right: 20px;" class="poster">
                        <div>
                            <h2>{selected_row['title']}<span style='color:gray; font-style:italic;'> ({selected_row['year']})</span></h2>
                            <div style='color:gray; font-style:italic;'>{get_tmdb_tagline(tmdb['id'])}</div>
                            <div class="genre">Genres: {', '.join(tmdb.get('genre_names', []))}</div> 
                            <div class="rating">⭐ {selected_row['averageRating']}/10</div><br>
                            <div class="description">{tmdb.get('overview', None)}</div>
                        </div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        # st.write("")

        col1, col2 = st.columns([2, 4])
        vid = get_tmdb_video(tmdb["id"])
        if vid and vid["site"] == "YouTube":
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(
                    f"""
                    <style>
                        .trailer-header {{
                            font-size: 28px; 
                            font-weight: bold;
                            color: #FF5722; 
                            margin-top: 20px; 
                            border-bottom: 3px solid #FF5722; 
                            padding-bottom: 10px; 
                            margin-bottom: 20px; 
                        }}
                    </style>
                    <div class="trailer-header">🎬 Bande annonce</div>
                """,
                    unsafe_allow_html=True,
                )
                st.video(f"https://www.youtube.com/watch?v={vid['key']}")
        else:
            with col1:
                st.markdown(
                    f"""
                    <style>
                        .trailer-header {{
                            font-size: 28px; 
                            font-weight: bold;
                            color: #FF5722; 
                            margin-top: 20px; 
                            border-bottom: 3px solid #FF5722; 
                            padding-bottom: 10px; 
                            margin-bottom: 20px; 
                        }}
                        .no-trailer {{
                            font-size: 18px;
                            color: #888888; /* Couleur grise pour le message d'absence */
                            margin-top: 10px;
                        }}
                    </style>
                    <div class="trailer-header">🎬 Bande annonce</div>
                    <div class="no-trailer">Pas de bande annonce disponible.</div>
                """,
                    unsafe_allow_html=True,
                )
        with col2:
            # Affichage des acteurs
            tmdb_actors = get_tmdb_actors(tmdb["id"])
            st.markdown(
                f"""
                <style>
                    .main-actors-title {{
                        font-size: 28px; 
                        font-weight: bold;
                        color: #4CAF50; /* Couleur verte */
                        margin-top: 20px; 
                        border-bottom: 3px solid #4CAF50; 
                        padding-bottom: 10px; 
                        margin-bottom: 20px; 
                    }}
                </style>
                <div class="main-actors-title">🎭 Acteurs principaux</div>
            """,
                unsafe_allow_html=True,
            )

            with st.container():
                cols = st.columns(len(tmdb_actors))  # une colonne par acteur
                for col, actor in zip(cols, tmdb_actors):
                    with col:
                        st.image(
                            f"https://media.themoviedb.org/t/p/w138_and_h175_face{actor['profile_path']}",
                            use_container_width=True,
                        )
                        if st.button(
                            f"{actor['name']}",
                            key=actor["name"],
                            use_container_width=True,
                        ):
                            st.session_state.movie_id = False
                            st.session_state.actor_id = actor["id"]
                            st.switch_page("pages/actor_details.py")

        "---"
        # Affichage de la liste des voisins
        st.markdown(
            f"""
            <style>
                .recommendations-title {{
                    font-size: 28px; /* Taille plus grande que Films associés */
                    font-weight: bold;
                    color: #2196F3; /* Couleur bleue */
                    margin-top: 20px; 
                    border-bottom: 3px solid #2196F3; 
                    padding-bottom: 10px; 
                    margin-bottom: 20px; 
                }}
            </style>
            <div class="recommendations-title">🔍 Recommandations associées :</div>
        """,
            unsafe_allow_html=True,
        )

        neighbors = find_neighbors_title(selected_row["title"], df)
        st.write("")
        st.dataframe(neighbors)  # Affichage du dataframe correspondant

    # Fonction pour afficher les plus proches voisins
    def find_neighbors_title(title, df):
        # films correspondant au titre donné
        filtered_df = df[df["title"].str.lower() == title.lower()]

        # Vérifie si le film existe dans le DataFrame
        if filtered_df.empty:
            st.write(f"Film '{title}' non trouvé.")
            return None

        # Index du 1er film avec le title donné
        index_imdb = filtered_df.index[0]

        # Voisins du film
        cv = CountVectorizer(stop_words="english")
        cv_matrix = cv.fit_transform(df["overview"])
        knn_model = NearestNeighbors(
            metric="cosine", algorithm="auto", n_jobs=-1, n_neighbors=4
        ).fit(cv_matrix)

        _, indices = knn_model.kneighbors(cv_matrix[index_imdb].reshape(1, -1))

        # Exclu le film lui-même des voisins
        filtered_indices = [i for i in indices[0] if i != index_imdb]

        # Retourner les voisins et leurs informations
        neighbors_df = df.iloc[filtered_indices]

        for _, row in neighbors_df.iterrows():
            tmdb = get_tmdb_data(row["imdb_id"])
            tag = get_tmdb_tagline(row["imdb_id"])
            with st.container():
                st.markdown(
                    f"""
                    <div class="movie-container">
                        <div style="display: flex; align-items: center;">
                            <img src="{row['poster_path']}" alt="Poster" style="width: 150px; margin-right: 20px;" class="poster">
                            <div>
                                <h3>{row['title']} <span style='color:gray; font-style:italic;'> ({row['year']})</span></h3>
                                <div style='color:gray; font-style:italic;'>{get_tmdb_tagline(row['imdb_id'])}</div>
                                <div class="genre">Genres: {', '.join(tmdb.get('genre_names', []))}</div> 
                                <div class="rating">⭐ {row['averageRating']}/10</div>
                                <div class="description">{tmdb.get('overview', None)}</div>
                            </div>
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            st.write("")

        return neighbors_df

    # Réinitialiser l'état des boutons lorsque l'on navigue entre les pages
    if "page" in st.session_state:
        if "plus" in st.session_state:
            st.session_state["plus"] = False
        # Réinitialiser tous les boutons des films (ayant des clés qui commencent par "movie_")
        for key in list(st.session_state.keys()):
            # Vérifiez si la clé commence par "movie_" et qu'elle n'est pas "movie_id" ou "plus"
            if key.startswith("movie_") and key not in ["movie_id", "plus"]:
                del st.session_state[key]

    # Saisie du film avec recherche systématique à chaque lettre tapée
    film = st_keyup("Saisissez un film", debounce=500, key="2")

    # Initialisation de movie_id si nécessaire
    if "movie_id" not in st.session_state:
        st.session_state.movie_id = None  # ou False, ou une autre valeur par défaut

    if film:
        if "movie_id" not in st.session_state:
            st.session_state.movie_id = False

        # Recherche des films correspondant au titre saisi
        results = search_movies(film, df)

        # Affiche les résultats
        if results.empty:
            st.write("Aucun film trouvé.")
        else:
            st.write(f"{len(results)} film(s) trouvé(s):")
            # Liste des films correspondants
            selected_movie = st.pills(
                "films",
                results["title"],
                selection_mode="single",
                default=None,
                label_visibility="hidden",
            )
            
            "---" # séparateur
            
            if selected_movie:
                # Trouver le film sélectionné dans le dataframe 'results'
                selected_row = results[
                    results["title"].str.lower() == selected_movie.lower()
                ].iloc[0]

                # Mettre en mémoire le film sélectionné
                st.session_state.movie_id = selected_row["imdb_id"]

                # Récupérer les données TMDb pour le film sélectionné
                tmdb = get_tmdb_data(selected_row["imdb_id"])

                # Affichage des informations détaillées du film
                display_movie_info(selected_row, tmdb)

    # Si un film est déjà sélectionné (par exemple après une recherche précédente)
    elif st.session_state.movie_id:
        # Récupération des données TMDb pour le film sélectionné
        tmdb = get_tmdb_data(st.session_state.movie_id)

        # Vérifier si le film existe dans le DataFrame
        if st.session_state.movie_id in df["imdb_id"].values:
            # Si le film est trouvé, récupérer la ligne correspondante
            selected_row = df[df["imdb_id"] == st.session_state.movie_id].iloc[0]

            # Affichage des informations détaillées du film
            display_movie_info(selected_row, tmdb)
        else:
            # Si le film n'est pas trouvé dans le DataFrame
            st.write(
                "Le film sélectionné n'existe plus ou a été supprimé de la base de données."
            )
            # Optionnellement, réinitialiser l'ID du film dans la session
            st.session_state.movie_id = None
