import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import requests
import re

MAIN_URL = "https://api.themoviedb.org/3"
API_KEY = '60c5ca9b75de2d2e768380e9a5bfd88c'

def get_details(actor_id):
    url = MAIN_URL + "/person/" + str(actor_id) + "?language=fr"
    params = { 'api_key': API_KEY }
    return requests.get(url, params=params).json()

def get_biography(actor_id):
    data = get_details(actor_id)
    return data["biography"]

def get_birthdate(actor_id):
    data = get_details(actor_id)
    return data["birthday"]

def get_movies(actor_id):
    url = MAIN_URL + "/person/" + str(actor_id) + "/movie_credits?language=fr"
    params = { 'api_key': API_KEY }
    movies = requests.get(url, params=params).json()['cast']
    list_movies = {}
    for movie in movies:
        if movie["release_date"]:
            year = re.search(r"\d{4}", movie["release_date"]).group(0)
        else:
            year = "Non renseigné"
        list_movies[movie["original_title"]] = year
    return list_movies

def list_movie_posters(actor_id):
    url = MAIN_URL + "/person/" + str(actor_id) + "/movie_credits?language=fr"
    params = { 'api_key': API_KEY }
    movies = requests.get(url, params=params).json()['crew']
    list_posters = []
    for m in movies:
      url = "https://image.tmdb.org/t/p/w500" + m["poster_path"]
      list_posters.append(url)
    return list_posters


