import streamlit as st

if "movie_id" not in st.session_state:
    st.session_state.movie_id = None
    
if st.query_params['movie_id']:   
    st.session_state.movie_id = st.query_params['movie_id']

st.switch_page("kpi.py")