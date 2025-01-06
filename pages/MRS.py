import streamlit as st
import pandas as pd
from st_keyup import st_keyup
import requests
from st_click_detector import click_detector 
# import streamlit.components.v1 as components

st.set_page_config(layout="wide")

API_KEY = '60c5ca9b75de2d2e768380e9a5bfd88c'

df = pd.read_parquet('imdb.parquet')

# Fonction de la page 2

st.header("Application de recommandations")

# Accéde au modèle NearestNeighbors depuis session_state
if 'nn_model' in st.session_state:
    nn = st.session_state.nn_model
else:
    # force la page a modelisation pour initier le model
    # st.session_state.page = pages[3]  
    st.write("pas de modele")
    
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

def get_tmdb_actors_list(tmdb_id):
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

def get_tmdb_actor_details(actor_id):    
    url = f'https://api.themoviedb.org/3/person/{actor_id}?language=fr-FR'
    params = {
        'api_key': API_KEY 
    }        
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data and len(data) > 0:
            return data
    return None



################################################################


movies_list = list(df["title"])

if "clicked" not in st.session_state:
    st.session_state["clicked"] = None

if "counter" not in st.session_state:
    st.session_state["counter"] = 1
    
if "movie_list" not in st.session_state:
    st.session_state["movie_list"] = movies_list
    
def get_index_from_titre(df: pd.DataFrame, titre: str) -> int:
    """
    Trouve l'index correspondant à un titre donné dans un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les titres.
    titre : str
        Le titre dont on cherche l'index.

    Returns
    -------
    int
        Index du titre dans le DataFrame.
    """
    return df[df.title == titre].index[0]

def afficher_top_genres(df: pd.DataFrame, genres: str) -> pd.DataFrame:
    """
    Affiche les films les mieux classés d'un genre spécifique, excluant "Animation" sauf si spécifié.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les films.
    genres : str
        Genre de films à afficher.

    Returns
    -------
    pd.DataFrame
        DataFrame des films triés par popularité, note moyenne, et nombre de votes.
    """
    sort_by = ["year", "averageRating", "numVotes"]
    ascending_ = [False for i in range(len(sort_by))]
    condi = (
        (
            df["genres"].str.contains(genres)
            & ~df["genres"].str.contains("Animation")
        )
        if genres != "Animation"
        else df["genres"].str.contains(genres)
    )
    return df[condi].sort_values(by=sort_by, ascending=ascending_)

def get_titre_from_index(df: pd.DataFrame, idx: int) -> str:
    """
    Récupère le titre correspondant à un index donné dans un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les titres.
    idx : int
        Index du titre à récupérer.

    Returns
    -------
    str
        Titre correspondant à l'index fourni.
    """
    return df[df.index == idx]["title"].values[0]

def get_info(df: pd.DataFrame, info_type: str):
    info = df[info_type].iloc[0]
    return info

def infos_button(df: pd.DataFrame, movie_list: list, idx: int):
    """
    Met à jour une variable de session Streamlit en fonction de l'index du film sélectionné.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les informations des films.
    movie_list : list
        Liste des titres de films.
    idx : int
        Index du film sélectionné.

    Cette fonction ne retourne rien mais met à jour la variable de session "index_movie_selected".
    """
    titre = get_titre_from_index(df, idx)
    st.session_state["index_movie_selected"] = movie_list.index(titre)
    
def get_info(df: pd.DataFrame, info_type: str):
    """
    Extrait une information spécifique du premier élément d'une colonne d'un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    info_type : str
        Le nom de la colonne dont on extrait l'information.

    Returns
    -------
    Any
        Information extraite de la première ligne de la colonne spécifiée.
    """
    info = df[info_type].iloc[0]
    return info

    
def afficher_details_film(df: pd.DataFrame, movies_ids: list):
    infos = {
        "id": get_info(df, "tmdb_id"),
        "date": get_info(df, "date"),
        "image": get_info(df, "image"),
        "titre_str": get_info(df, "titre_str"),
        "titre_genres": get_info(df, "titre_genres"),
        "rating_avg": round(get_info(df, "rating_avg"), 1),
        "rating_vote": get_info(df, "rating_vote"),
        "popularity": get_info(df, "popularity"),
        "runtime": get_info(df, "runtime"),
        "synopsis": get_info(df, "overview"),
        "tagline": get_info(df, "tagline"),
        "youtube": get_info(df, "youtube"),
    }
    st.write("llllllllllllllllllllllllllllllllll")
    film_str: str = infos["titre_str"]
    name_film = film_str[:-7] if film_str.endswith(")") else film_str
    runtime = infos["runtime"]
    # actors_list = [a for a in get_actors_dict(df).values()]
    # director_list = [d for d in get_directors_dict(df).values()]
    # director = asyncio.run(
    #     fetch_persons_bio(director_list, movies_ids, True)
    # )
    # actors = asyncio.run(fetch_persons_bio(actors_list, movies_ids))

    col1, col2, cols3 = st.columns([1, 2, 1])
    with col1:
        st.image(infos["image"], use_container_width=True)
    with col2:
        st.header(
            f"{name_film} - ({infos['date']})", anchor=False, divider=True
        )

        st.caption(
            f"<p style='font-size: 16px;'>{infos['titre_genres']} • {f'{runtime // 60}h {runtime % 60}m'}</p>",
            unsafe_allow_html=True,
        )
        texte_fondu = (
            f'<span style="color: #555;">*"{infos["tagline"]}"*</span>'
        )
        st.write(texte_fondu, unsafe_allow_html=True)
        color_rating = (
            "#198c19"
            if infos["rating_avg"] >= 7
            else "#F4D03F"
            if infos["rating_avg"] >= 5
            else "#E74C3C"
        )
        txt_color = "#F2F2F2" if infos["rating_avg"] >= 7 else "#191919"

        gap = 0.1

        elements_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; gap: {gap}px;">
                <p>Notes</p>
                <div style="background-color: {color_rating}; border-radius: 50%; width: 60px; height: 60px;">
                    <h2 style="text-align: center; color: {txt_color}; font-size: 22px;">{round(infos["rating_avg"], 2)}</h2>
                </div>
            </div>
        """
        st.markdown(
            f"<div style='display: flex; justify-content: start; gap: 20px;'>{elements_html}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        full_perso = director + actors
        cols = st.columns(len(full_perso))
        actors_ids = [n["id"] for n in actors]
        character = asyncio.run(
            fetch_persons_movies(infos["id"], actors_ids)
        )

        for i, col in enumerate(cols):
            # st.session_state["person_id"] = full_perso[i]["id"]

            with col:
                if i < 1:
                    st.subheader(
                        "**Réalisation**", anchor=False, divider=True
                    )
                elif i == len(director):
                    st.subheader("**Casting**", anchor=False, divider=True)
                else:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                prso_dict, clicked2 = get_clicked_act_dirct(
                    full_perso, character, i
                )
                if clicked2:
                    st.session_state["clicked2"] = True
                    st.session_state["actor"] = prso_dict
        if st.session_state["clicked2"]:
            switch_page("full_bio")
            st.session_state["counter"] += 1
            auto_scroll()
            st.rerun()

    with cols3:
        st.header("**Bande Annonce** ", anchor=False, divider=True)
        youtube_url = (
            str(infos["youtube"]).replace("watch?v=", "embed/")
            + "?autoplay=0&mute=1"
        )
        yout = f"""
            <div style="margin-top: 20px;">
                <iframe width="100%" height="315" src="{youtube_url}" frameborder="0" allowfullscreen></iframe>
            </div>
        """
        # st.video(infos["youtube"])
        st.markdown(yout, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    
def get_clicked(
    df: pd.DataFrame,
    titres_list: list,
    nb: int,
    genre: str = "Drame",
    key_: bool = False,
):
    """
    Génère un élément cliquable pour un film et renvoie son index et un détecteur de clic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les films.
    titres_list : list
        Liste des titres de films.
    nb : int
        Numéro du film dans la liste.
    genre : str, optional
        Genre du film, par défaut à "Drame".
    key_ : bool, optional
        Si vrai, génère une clé unique pour le détecteur de clic, par défaut à False.

    Returns
    -------
    Tuple[int, Any]
        Index du film et l'objet du détecteur de clic.
    """
    index = int(get_index_from_titre(df, titres_list[nb]))
    movie = df[df["title"] == titres_list[nb]]
    image_link = get_info(movie, "poster_path")
    titile = get_info(movie, "title")
    # content = f"""
    #     <div style="text-align: center;">
    #         <a href="#" id="{titres_list[nb]}">
    #             <img width="125px" heigth="180px" src="{image_link}"
    #                 style="object-fit: cover; border-radius: 5%; margin-bottom: 15px;">
    #         </a>
    #         <p style="margin: 0;">{titile}</p>
    # """
    content = f"""
        <div style="text-align: center;">
            <a href="#" id="{titres_list[nb]}">
                <img width="125px" height="180px" src="{image_link}"
                    style="object-fit: cover; border-radius: 5%; margin-bottom: 15px; cursor: pointer; transition: filter .2s ease-in-out, transform .2s ease-in-out;"
                    onmouseover="this.style.filter='brightness(70%)'; this.style.transform='scale(1.1)'"
                    onmouseout="this.style.filter='brightness(100%)'; this.style.transform='scale(1)'">
            </a>
            <p style="margin: 0;">{titile}</p>
        </div>
    """
    
    if key_:
        unique_key = f"click_detector_{genre}_{index}"
        return index, click_detector(content, key=unique_key)
    else:
        return index, click_detector(content)
    
    
st.write(df.columns)    
genres_list = [
    "Action"
]
for genre in genres_list:
    genre_df = afficher_top_genres(df, genre)
    titres = genre_df["title"].head(10).tolist()
    st.header(f"Top 10 {genre}", anchor=False)
    cols = st.columns(10)
    for i, col in enumerate(cols):
        with col:
            index, clicked = get_clicked(
                genre_df, titres, i, genre, True
            )
            if clicked:
                st.session_state["clicked"] = index
                st.write(index)
    if st.session_state["clicked"] is not None:
        infos_button(df, movies_list, st.session_state["clicked"])
        st.session_state["counter"] += 1
        # auto_scroll()
        st.rerun()  
    
    
      
      
      
      
      
      ##############################################################################################################
      
      
      
      
      
      
      
      
      
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
            tmdb_actors = get_tmdb_actors_list(tmdb['id'])
            st.header("Acteurs principaux")
            html_code = """
                <a id='{actor_id}'>
                <div id="custom-element-{actor_id}" style="text-align: center; padding: 10px;">
                    <img src="https://media.themoviedb.org/t/p/w138_and_h175_face{profile_path}" alt="{actor_name}" style="width: 100%; max-width: 150px;"/>
                    <p style='margin: 0px;
                        font-family: "Source Sans Pro", sans-serif;
                        font-weight: 400;
                        line-height: 1.6;
                        color: rgb(250, 250, 250);
                        background-color: rgb(14, 17, 23);
                        text-size-adjust: 100%;
                        -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
                        -webkit-font-smoothing: auto;'>
                        {actor_name}
                    </p>
                </div></a>
            """

            with st.container():
                cols = st.columns(len(tmdb_actors))  # une colonne par acteur
                for col, actor in zip(cols, tmdb_actors):
                    with col:
                        actor_html = html_code.format(profile_path=actor['profile_path'], actor_name=actor['name'], actor_id=actor['id'])
                        # components.html(actor_html, height=250)            
                        
                                   
                                   
                    #     st.write(actor_html)
                    #     actor_clicked = click_detector(actor_html, key=str(actor['id']))
                    # if actor_clicked:
                    #     st.write(actor_clicked)
                    #     details = get_tmdb_actor_details(actor_clicked)
                    #     st.write(details)
                        
                    #     @st.dialog(title=details['name'], width="large")
                    #     def affiche_details():
                    #         st.write(details['place_of_birth'])
                    #         if details['biography']:
                    #             st.write(details['biography'])
                    #     affiche_details()

                       
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        # st.markdown(f"**{clicked} clicked**" if clicked != "" else "")            
                        
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
    
st.sidebar.page_link('kpi.py', label='KPI')
st.sidebar.page_link('pages/MRS.py', label='Recommandations de films')

