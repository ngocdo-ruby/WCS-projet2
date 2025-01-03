import streamlit as st
import requests
from datetime import datetime

def actor_details():
    API_KEY = "60c5ca9b75de2d2e768380e9a5bfd88c"
    BASE_URL = "https://api.themoviedb.org/3"
    ACTOR_ID = st.session_state.actor_id

    # Fonction pour obtenir les informations sur l'acteur
    def get_actor_info(actor_id):
        url = f"{BASE_URL}/person/{actor_id}?api_key={API_KEY}&language=fr"
        response = requests.get(url)
        return response.json()

    # Fonction pour obtenir les films associés à l'acteur
    def get_actor_movies(actor_id):
        url = (
            f"{BASE_URL}/person/{actor_id}/movie_credits?api_key={API_KEY}&language=fr"
        )
        response = requests.get(url)
        return response.json()

    # Fonction pour obtenir les images de l'acteur
    def get_actor_images(actor_id):
        url = f"{BASE_URL}/person/{actor_id}/images?api_key={API_KEY}"
        response = requests.get(url)
        return response.json()

    # Fonction pour obtenir les id reseaux sociaux
    def get_actor_reseaux(actor_id):
        url = f"{BASE_URL}/person/{actor_id}/external_ids?api_key={API_KEY}"
        response = requests.get(url)
        return response.json()

    # Fonction pour obtenir des détails supplémentaires sur un film
    def get_movie_details(movie_id):
        url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=fr"
        response = requests.get(url)
        return response.json()

    # Fonction de gestion d'un bouton avec état
    def stateful_button(*args, key=None, **kwargs):
        if key is None:
            raise ValueError("Must pass key")

        # Si la clé n'existe pas dans session_state, on l'initie
        if key not in st.session_state:
            st.session_state[key] = False

        # Si le bouton est cliqué, on inverse l'état
        if st.button(*args, **kwargs):
            st.session_state[key] = not st.session_state[key]

        return st.session_state[key]

    # Réinitialiser l'état des boutons lorsque l'on navigue entre les pages
    if "page" in st.session_state:
        # Réinitialiser tous les boutons des films (clés qui commencent par "movie_")
        for key in list(st.session_state.keys()):
            # Vérifiez si la clé commence par "movie_" et qu'elle n'est pas "movie_id"
            if key.startswith("movie_") and key not in ["movie_id"]:
                st.session_state[key] = False

    # Informations de l'acteur
    actor_info = get_actor_info(ACTOR_ID)
    actor_movies = get_actor_movies(ACTOR_ID)
    actor_images = get_actor_images(ACTOR_ID)
    actor_reseaux = get_actor_reseaux(ACTOR_ID)

    # Mise en page de colonnes pour le profil de l'acteur
    col1, col2 = st.columns([1, 4])

    # Colonne de gauche : Image de profil
    with col1:
        if actor_info["profile_path"]:
            st.image(f"https://image.tmdb.org/t/p/w500{actor_info['profile_path']}")
        else:
            st.image("https://via.placeholder.com/200", width=200)

        # Infos de l'acteurs
        homepage = actor_info["homepage"]
        instagram_url = (
            f"https://instagram.com/{actor_reseaux['instagram_id']}/"
            if actor_reseaux["instagram_id"]
            else ""
        )
        birth_date = datetime.strptime(actor_info["birthday"], "%Y-%m-%d")
        death_date = (datetime.strptime(actor_info["deathday"], "%Y-%m-%d") if actor_info["deathday"] else "")
        # Format de la date en "dd/mm/yyyy"
        formatted_birth_date = birth_date.strftime("%d/%m/%Y")
        formatted_death_date = (death_date.strftime("%d/%m/%Y") if death_date else "")
        # Calcul de l'âge
        age = (death_date.year - birth_date.year - ((death_date.month, death_date.day) < (birth_date.month, birth_date.day)) if death_date else datetime.today().year - birth_date.year - ((datetime.today().month, datetime.today().day) < (birth_date.month, birth_date.day)))
        birth_place = actor_info["place_of_birth"]
        # nombre d'apparition
        known_appearances = len(actor_movies["cast"])
        # genre
        gender = (
            "Homme"
            if actor_info["gender"] == 2
            else "Femme"
            if actor_info["gender"] == 1
            else ""
        )

        # Galerie photos
        if actor_images.get("profiles"):
            cols = st.columns(5)  # 5 colonnes pour afficher les images horizontalement
            for i, image in enumerate(
                actor_images["profiles"][1:6]
            ):  # Afficher les 5 premières images sauf celle deja utilisée en principale
                with cols[i]:
                    st.image(
                        f"https://image.tmdb.org/t/p/w500{image['file_path']}",
                        width=150,
                    )
        else:
            st.write("Aucune photo disponible.")

    # Colonne de droite : Informations de l'acteur
    with col2:
        st.title(actor_info["name"])
        st.write(f"**Biographie** :")
        st.write(f"{actor_info['biography'][:2000]}...")  # Affichage limité de la bio

    st.write("")

    # Mise en page 2eme ligne (infos personnelles + films associés)
    col3, col4 = st.columns([1, 4])
    with col3:
        full_wrapper_html = """
            <style>
                .full_wrapper {
                    background-color: #e8f5e9;
                    display: block;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                .reseaux {
                    margin-bottom: 10px;
                    text-decoration: none;
                    color: #3b5998;
                    font-size: 16px;
                }
                h3 {
                    font-size: 18px;
                    margin-bottom: 10px;
                    color: #333;
                }
                .infos_perso p {
                    margin: 5px 0;
                    color: #555;
                    font-size: 14px;
                }
            </style>
        """
        full_wrapper_html += f"""<div class="full_wrapper">"""

        # Ajout du lien homepage si défini
        if homepage:
            full_wrapper_html += f"""<div class="reseaux"><a class="reseaux" href="{homepage}" target="_blank" rel="noopener">🌐 Homepage</a></div>"""

        # Ajout du lien Instagram si défini
        if instagram_url:
            full_wrapper_html += f"""<div class="reseaux"><a class="reseaux" href="{instagram_url}" target="_blank" rel="noopener">📷 Instagram</a></div>"""

        # Ajout de la section des informations personnelles si les données sont définies
        full_wrapper_html += """
        <h3>Informations personnelles</h3>
            <div class="infos_perso">
        """

        # Ajout des informations personnelles, si elles existent
        if known_appearances:
            full_wrapper_html += (
                f"<p><strong>Apparitions connues :</strong> {known_appearances}</p>"
            )
        if gender:
            full_wrapper_html += f"<p><strong>Genre :</strong> {gender}</p>"
        if death_date:
            full_wrapper_html += f"<p><strong>Date de naissance :</strong> {formatted_birth_date}</p>"
            full_wrapper_html += f"<p><strong>Date de décès :</strong> {formatted_death_date} ({age} ans)</p>"
        elif birth_date:
            full_wrapper_html += f"<p><strong>Date de naissance :</strong> {formatted_birth_date} ({age} ans)</p>"
        if birth_place:
            full_wrapper_html += (
                f"<p><strong>Lieu de naissance :</strong> {birth_place}</p>"
            )

        # Fermeture de la section "Informations personnelles"
        full_wrapper_html += "</div>"

        # Fermeture du wrapper global
        full_wrapper_html += "</div>"

        # Affichage du HTML avec les styles et conditions
        st.markdown(full_wrapper_html, unsafe_allow_html=True)

    with col4:
        # Films associés
        st.markdown(
            """
            <style>
                .section-title {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50; 
                    border-bottom: 2px solid #4CAF50; 
                    padding-bottom: 5px;
                    margin-bottom: 15px;
                }
            </style>
            <div class="section-title">🎬 Films associés</div>
        """,
            unsafe_allow_html=True,
        )

        cols = st.columns(5)  # 5 colonnes pour afficher les films horizontalement
        for i, movie in enumerate(
            actor_movies["cast"][:5]
        ):  # Afficher les 5 premiers films
            movie_title = movie["title"]
            release_date = movie.get("release_date", "Date inconnue")[:4]
            movie_id = movie["id"]

            # Détails du film
            movie_details = get_movie_details(movie_id)

            with cols[i]:
                st.image(
                    f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}",
                    use_container_width=True,
                )
                # st.write(f"**Résumé**: {movie_details['overview'][:100]}...")  # Affichage d'un résumé limité
                if stateful_button(
                    f"### {movie_title} ({release_date})",
                    key=f"movie_{movie_id}",
                    use_container_width=False,
                ):
                    st.session_state.movie_id = movie_details["imdb_id"]
                    st.switch_page("kpi.py")

    # Section d'autres films (facultatif) : pour la pagination ou plus de films
    if len(actor_movies["cast"]) > 5:
        # Si on clique sur "Afficher plus de films"
        if stateful_button("Afficher plus de films", key="plus", type="primary"):
            cols = st.columns(
                5
            )  # Recréation de 5 colonnes pour afficher les films supplémentaires
            for i, movie in enumerate(
                [
                    movie
                    for movie in actor_movies["cast"]
                    if movie.get("poster_path") is not None
                ][5:15]
            ):  # Afficher le reste des films
                movie_title = movie["title"]
                release_date = movie.get("release_date", "")[:4]
                movie_id = movie["id"]

                # Détails du film
                movie_details = get_movie_details(movie_id)

                with cols[i % 5]:  # S'assure qu'on reste dans les 5 colonnes
                    # st.markdown(f"### {movie_title} ({release_date})")
                    st.image(
                        f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}",
                        use_container_width=True,
                    )
                    if stateful_button(
                        f"### {movie_title} ({release_date})",
                        key=f"movie_{movie_id}",
                        use_container_width=True,
                    ):
                        st.session_state.movie_id = movie_details["imdb_id"]
                        st.switch_page("kpi.py")

    # Sidebar pour naviguer
    st.sidebar.page_link("kpi.py", label=f"🎬 Accueil")
    st.sidebar.page_link("pages/actor_details.py", label=f"🎭 {actor_info['name']}")

if __name__ == "__main__":
    actor_details()
