import streamlit as st
import os
import re
import time
import pickle
import requests
import pandas as pd
import tmdbsimple as tmdb
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Movie Chooser IA", page_icon="🍿", layout="centered")

# --- ⚠️ TES INFORMATIONS ⚠️ ---
tmdb.API_KEY = '01c6510a9d1383e8e5adbbc54a69bfaf' 
PSEUDO_LETTERBOXD = 'Clement_Blld'
CHEMIN_CACHE = 'bibliotheque_ia_cache.pkl' 
# -------------------------------------

GENRES_TMDB = {
    "action": 28, "aventure": 12, "animation": 16, "comédie": 35, "comedie": 35,
    "crime": 80, "policier": 80, "documentaire": 99, "drame": 18, "famille": 10751,
    "fantastique": 14, "histoire": 36, "historique": 36, "horreur": 27, "musique": 10402,
    "mystère": 9648, "romance": 10749, "romantique": 10749, "sf": 878,
    "science-fiction": 878, "espace": 878, "thriller": 53, "guerre": 10752,
    "western": 37
}

# --- FONCTION DE CHARGEMENT ET MISE À JOUR ---
@st.cache_data(ttl=3600, show_spinner="Synchronisation avec Letterboxd...") 
def preparer_bibliotheque():
    URL_RSS = f"https://letterboxd.com/{PSEUDO_LETTERBOXD}/rss/"
    ma_bibliotheque = []
    
    # 1. On charge ton fichier .pkl uploadé sur GitHub
    if os.path.exists(CHEMIN_CACHE):
        with open(CHEMIN_CACHE, 'rb') as f:
            df_biblio_full = pickle.load(f)
        ma_bibliotheque = df_biblio_full.to_dict('records')
    else:
        st.error("❌ Fichier bibliotheque_ia_cache.pkl introuvable sur le GitHub.")
        return None, None

    # 2. On vérifie les nouveautés via RSS
    try:
        r = requests.get(URL_RSS)
        soup = BeautifulSoup(r.content, "xml")
        titres_en_cache = [str(film['titre']).lower().strip() for film in ma_bibliotheque]
        films_a_ajouter = []
        
        for item in soup.find_all("item"):
            title_tag = item.find("letterboxd:filmTitle")
            year_tag = item.find("letterboxd:filmYear")
            rating_tag = item.find("letterboxd:memberRating")
            
            if title_tag and rating_tag:
                titre = title_tag.text
                if titre.lower().strip() not in titres_en_cache:
                    films_a_ajouter.append({"Name": titre, "Year": year_tag.text if year_tag else "", "Rating": float(rating_tag.text)})
                    
        # 3. Traitement des nouveautés à la volée
        if len(films_a_ajouter) > 0:
            for film in films_a_ajouter:
                try:
                    search = tmdb.Search()
                    search.movie(query=film['Name'], year=film['Year'])
                    if search.results:
                        info = tmdb.Movies(search.results[0]['id']).info(language='fr-FR')
                        kw = " ".join([k['name'] for k in tmdb.Movies(search.results[0]['id']).keywords().get('keywords', [])])
                        ma_bibliotheque.append({'titre': film['Name'], 'metadata': f"{info.get('overview', '')} {info.get('genres', '')} {kw}", 'ma_note': float(film['Rating'])})
                except: continue
    except:
        pass # Si Letterboxd bug, on utilise quand même le cache existant

    liste_deja_vus = [str(film['titre']).lower().strip() for film in ma_bibliotheque]
    return pd.DataFrame(ma_bibliotheque), liste_deja_vus


# --- LE MOTEUR DE RECOMMANDATION ---
def moteur_recommandation(demande_genres, demande_dates, demande_acteur, df_biblio, liste_deja_vus):
    demande_genres_lower = demande_genres.lower()
    ids_trouves = [str(GENRES_TMDB[mot]) for mot in GENRES_TMDB if mot in demande_genres_lower]
    genres_str = ",".join(ids_trouves)
    
    demande_dates_lower = demande_dates.lower()
    date_gte, date_lte = None, None
    if re.search(r'ann[ée]es?\s*(19|20)?(70|80|90|00|10|20)', demande_dates_lower):
        if "70" in demande_dates_lower: date_gte, date_lte = "1970-01-01", "1979-12-31"
        elif "80" in demande_dates_lower: date_gte, date_lte = "1980-01-01", "1989-12-31"
        elif "90" in demande_dates_lower: date_gte, date_lte = "1990-01-01", "1999-12-31"
        elif "2000" in demande_dates_lower or " 00" in demande_dates_lower: date_gte, date_lte = "2000-01-01", "2009-12-31"
        elif "2010" in demande_dates_lower: date_gte, date_lte = "2010-01-01", "2019-12-31"
        elif "2020" in demande_dates_lower: date_gte, date_lte = "2020-01-01", "2029-12-31"
    elif re.search(r'\b(19\d{2}|20\d{2})\b', demande_dates_lower):
        annee = re.search(r'\b(19\d{2}|20\d{2})\b', demande_dates_lower).group(1)
        date_gte, date_lte = f"{annee}-01-01", f"{annee}-12-31"
    elif re.search(r'\b(70|80|90)\b', demande_dates_lower):
        if "70" in demande_dates_lower: date_gte, date_lte = "1970-01-01", "1979-12-31"
        elif "80" in demande_dates_lower: date_gte, date_lte = "1980-01-01", "1989-12-31"
        elif "90" in demande_dates_lower: date_gte, date_lte = "1990-01-01", "1999-12-31"

    acteur_id = None
    if demande_acteur.strip():
        res_person = tmdb.Search().person(query=demande_acteur.strip())
        if res_person['results']:
            acteur_id = res_person['results'][0]['id']
        else:
            return f"<p style='color:red;'>⚠️ Aucun acteur nommé '{demande_acteur}' trouvé.</p>"

    discover = tmdb.Discover()
    candidats = []
    params = {'language': 'fr-FR', 'sort_by': 'popularity.desc', 'vote_count_gte': 50 if acteur_id else 100} 
    
    if genres_str: params['with_genres'] = genres_str
    if date_gte:
        params['primary_release_date.gte'] = date_gte
        params['primary_release_date.lte'] = date_lte
    if acteur_id:
        params['with_cast'] = str(acteur_id)
        
    for page in range(1, 4):
        params['page'] = page
        res = discover.movie(**params)
        for film in res['results']:
            if film.get('title', '').lower().strip() in liste_deja_vus or film.get('original_title', '').lower().strip() in liste_deja_vus: continue
            try:
                info = tmdb.Movies(film['id']).info(language='fr-FR')
                kw = " ".join([k['name'] for k in tmdb.Movies(film['id']).keywords().get('keywords', [])])
                candidats.append({
                    'titre': info['title'],
                    'date': info.get('release_date', 'Inconnue')[:4],
                    'desc': f"{info.get('overview', '')} {info.get('genres', '')} {kw}",
                    'resume_court': info.get('overview', 'Pas de résumé.')[:200] + "...",
                    'note': info['vote_average'],
                    'poster': f"https://image.tmdb.org/t/p/w200{film['poster_path']}" if film.get('poster_path') else "https://via.placeholder.com/200x300?text=Pas+d'affiche"
                })
            except: continue

    if not candidats: return f"<p>⚠️ Aucun film inédit trouvé avec ces critères !</p>"
        
    df_cand = pd.DataFrame(candidats)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_biblio['metadata'].tolist() + df_cand['desc'].tolist())
    v_biblio, v_cand = tfidf_matrix[:len(df_biblio)], tfidf_matrix[len(df_biblio):]
    
    df_cand['affinité_ia'] = [sum(cosine_similarity(v_cand[i], v_biblio)[0] * (df_biblio['ma_note'] - 2.5)) for i in range(v_cand.shape[0])]
    df_cand['score_final'] = df_cand['affinité_ia'] + (df_cand['note'] * 0.2)
    df_final = df_cand.sort_values(by='score_final', ascending=False).head(10)
    
    html = "<div style='display: flex; flex-direction: column; gap: 15px; margin-top:10px;'>"
    for _, row in df_final.iterrows():
        html += f"""
        <div style='display: flex; background: #222; color: white; border-radius: 8px; overflow: hidden; font-family: sans-serif; box-shadow: 0 4px 8px rgba(0,0,0,0.3);'>
            <img src='{row['poster']}' style='width: 100px; object-fit: cover;' />
            <div style='padding: 15px;'>
                <h3 style='margin: 0 0 5px 0; color: #4db8ff; font-size: 1.2em;'>{row['titre']} <span style='color:#aaa; font-size: 0.8em;'>({row['date']})</span></h3>
                <p style='margin: 0 0 10px 0; font-size: 13px; color: #ccc;'>
                    🏆 <b>Score Global : {row['score_final']:.2f}</b>  
                    <span style='color:#888; margin-left:10px;'>(IA: {row['affinité_ia']:.2f} + Public: {row['note']}/10)</span>
                </p>
                <p style='margin: 0; font-size: 12px; line-height: 1.4; color: #ddd;'>{row['resume_court']}</p>
            </div>
        </div>
        """
    html += "</div>"
    return html

# --- INTERFACE UTILISATEUR STREAMLIT ---
st.title("🎬 Ton Conseiller Cinéma IA")
st.markdown("Trouve le film parfait basé sur **ton ADN Letterboxd**.")

df_biblio_full, liste_deja_vus = preparer_bibliotheque()

if df_biblio_full is not None:
    # Les 3 champs de recherche alignés
    col1, col2, col3 = st.columns(3)
    with col1:
        text_genres = st.text_input("🎭 Genre :", placeholder="Ex: SF, Thriller...")
    with col2:
        text_dates = st.text_input("📅 Période :", placeholder="Ex: années 2000...")
    with col3:
        text_acteur = st.text_input("🌟 Acteur :", placeholder="Ex: Brad Pitt")
    
    # Le bouton
    if st.button("🎥 Trouver mon film", type="primary", use_container_width=True):
        if text_genres.strip() or text_dates.strip() or text_acteur.strip():
            with st.spinner("🍿 L'IA analyse la base de données..."):
                html_result = moteur_recommandation(text_genres, text_dates, text_acteur, df_biblio_full, liste_deja_vus)
                st.markdown(html_result, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Veuillez remplir au moins l'une des cases !")
