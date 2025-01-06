import streamlit as st

# Fonction de la page 2
def page_2():
    st.title("Page 2")
    
    st.header("Objectif")

    st.write("Réaliser une analyse approfondie de la base de données pour identifier des tendances et caractéristiques spécifiques.")
    st.write("Cette analyse devrait inclure : l’identification des acteurs les plus présents et les périodes associées, l’évolution de la durée moyenne des films au fil des années, la comparaison entre les acteurs présents au cinéma et dans les séries, l’âge moyen des acteurs, ainsi que les films les mieux notés et les caractéristiques qu’ils partagent.")

    st.write("Sur la base des informations récoltées, vous pourrez affiner votre programmation en vous spécialisant par exemple sur les films des années 90 ou les genres d’action et d’aventure, afin de mieux répondre aux attentes du public identifié lors de l’étude de marché")

    st.header("Besoin client")

    st.write("Obtenir quelques statistiques sur les films (type, durée), acteurs (nombre de films, type de films) et d’autres.")
    
st.sidebar.page_link('kpi.py', label='KPI')
st.sidebar.page_link('pages/prMRSojet.py', label='Recommandations de films')

if __name__ == "__main__":
    page_2()