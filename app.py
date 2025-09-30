import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt', download_dir='./nltk_data')
from nltk import tokenize
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px

# Nuovi import per Google APIs e utilità
import os
import re
import json
import tempfile
from typing import List, Tuple
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from google.auth.transport.requests import Request
from urllib.parse import urlencode

supported_file_types = ["docx", "pdf", "txt", "java"]

# Scopes: Classroom lettura compiti/consegne e Drive read-only
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/classroom.coursework.students.readonly",
    "https://www.googleapis.com/auth/classroom.student-submissions.students.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile"
]

# --------------------- Helper Google OAuth & Classroom/Drive ---------------------
def _persist_client_secret(uploaded_json_file) -> str:
    """Persisti il client_secret.json caricato e restituisci il path locale."""
    target_dir = os.path.join(".streamlit")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "google_client_secret.json")
    with open(target_path, "wb") as f:
        f.write(uploaded_json_file.getbuffer())
    return target_path

def build_codespaces_redirect_uri(port: int | None = None) -> str | None:
    """Costruisce l'URL pubblico del Codespace per la porta indicata.
    Usa CODESPACE_NAME e GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN.
    """
    cs = os.getenv("CODESPACE_NAME")
    domain = os.getenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
    if port is None:
        port = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    if cs and domain:
        return f"https://{cs}-{port}.{domain}"
    return None

def get_google_credentials(client_secret_path: str) -> Credentials:
    """OAuth flow compatibile con Codespaces: usa redirect all'URL pubblico del Codespace.
    Se non in Codespaces, fallback al server locale.
    """
    # Token in sessione o su disco
    if "google_creds" in st.session_state:
        creds = st.session_state["google_creds"]
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state["google_creds"] = creds
            return creds
    creds = None
    token_path = os.path.join(".streamlit", "token.json")
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, GOOGLE_SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                st.session_state["google_creds"] = creds
                return creds
        except Exception:
            creds = None

    # Se non abbiamo credenziali valide, avvia o completa il flow
    redirect_uri = build_codespaces_redirect_uri()
    params = dict(st.query_params)
    if redirect_uri:
        # Se siamo tornati da Google con ?code=..., completa il flow ricostruendo un Flow con lo state
        if 'code' in params and 'state' in params:
            try:
                flow = Flow.from_client_secrets_file(
                    client_secrets_file=client_secret_path,
                    scopes=GOOGLE_SCOPES,
                    redirect_uri=redirect_uri,
                    state=params.get('state')
                )
                authorization_response = redirect_uri.rstrip('/') + '?' + urlencode(params)
                flow.fetch_token(authorization_response=authorization_response)
                creds = flow.credentials
                st.session_state['google_creds'] = creds
                # persisti su file
                os.makedirs(".streamlit", exist_ok=True)
                with open(token_path, "w") as token:
                    token.write(creds.to_json())
                # pulisci query params per evitare ri-esecuzioni
                try:
                    st.query_params.clear()
                except Exception:
                    try:
                        st.experimental_set_query_params()
                    except Exception:
                        pass
                return creds
            except Exception as e:
                st.error(f"Errore nella finalizzazione OAuth: {e}")
        # Altrimenti avvia il flow e mostra il link per proseguire
        flow = Flow.from_client_secrets_file(client_secret_path, scopes=GOOGLE_SCOPES, redirect_uri=redirect_uri)
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        # Memorizzare lo state non è più necessario per la finalizzazione, ma lo teniamo per debug
        st.session_state['oauth_state'] = state
        st.markdown(f"[Continua l'autenticazione Google]({auth_url})")
        st.stop()
    else:
        # Fallback: ambiente locale, usa server locale
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, GOOGLE_SCOPES)
        creds = flow.run_local_server(port=0, prompt='consent', authorization_prompt_message='')
        os.makedirs(".streamlit", exist_ok=True)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
        st.session_state['google_creds'] = creds
        return creds

def parse_coursework_url(url: str) -> Tuple[str, str]:
    """Estrai course_id e course_work_id da un URL tipo
    https://classroom.google.com/c/<course_id>/a/<course_work_id>/details
    """
    m = re.search(r"/c/([^/]+)/a/([^/]+)/?", url)
    if not m:
        raise ValueError("URL di Classroom non valido. Atteso formato .../c/<course_id>/a/<course_work_id>/...")
    return m.group(1), m.group(2)

def classroom_service(creds: Credentials):
    return build('classroom', 'v1', credentials=creds)

def drive_service(creds: Credentials):
    return build('drive', 'v3', credentials=creds)

def list_student_submissions(classroom, course_id: str, course_work_id: str) -> List[dict]:
    submissions = []
    page_token = None
    while True:
        resp = classroom.courses().courseWork().studentSubmissions().list(
            courseId=course_id,
            courseWorkId=course_work_id,
            pageToken=page_token
        ).execute()
        submissions.extend(resp.get('studentSubmissions', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return submissions

def get_user_surname(classroom, user_id: str) -> str:
    try:
        profile = classroom.userProfiles().get(userId=user_id).execute()
        name = profile.get('name', {})
        return name.get('familyName') or name.get('fullName') or str(user_id)
    except Exception:
        return str(user_id)

def _download_drive_file_as_bytes(drive, file_id: str, export_mime: str = None) -> bytes:
    buf = io.BytesIO()
    if export_mime:
        request = drive.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = drive.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()

def drive_file_to_text(drive, file_id: str, mime_type_hint: str = None) -> str:
    """Scarica un file di Drive e restituisce testo. Per Google Docs usa export text/plain."""
    meta = drive.files().get(fileId=file_id, fields="id, name, mimeType").execute()
    mime = meta.get('mimeType')
    name = meta.get('name', '')

    # Google Docs/Slides/Sheets -> export
    if mime and mime.startswith('application/vnd.google-apps'):
        # Solo Docs può essere esportato come testo semplice in maniera affidabile
        export_mime = 'text/plain'
        data = _download_drive_file_as_bytes(drive, file_id, export_mime=export_mime)
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return data.decode('latin-1', errors='ignore')

    # File non-google: scarica contenuto binario e processa in base al mime
    data = _download_drive_file_as_bytes(drive, file_id)
    # Se è testo
    if mime and (mime.startswith('text/') or mime in ('application/json',)):
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return data.decode('latin-1', errors='ignore')
    # PDF
    if mime == 'application/pdf' or (name.lower().endswith('.pdf')):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception:
            return ""
    # DOCX
    if mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or name.lower().endswith('.docx'):
        try:
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=True) as tmp:
                tmp.write(data)
                tmp.flush()
                return docx2txt.process(tmp.name) or ""
        except Exception:
            return ""
    # Prova come utf-8 generico
    try:
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ""

def extract_texts_from_submissions(creds: Credentials, course_id: str, course_work_id: str) -> Tuple[List[str], List[str]]:
    """Per ogni consegna, concatena il testo degli allegati driveFile e usa il cognome come nome file."""
    cls = classroom_service(creds)
    drv = drive_service(creds)
    submissions = list_student_submissions(cls, course_id, course_work_id)
    texts: List[str] = []
    names: List[str] = []

    for sub in submissions:
        user_id = sub.get('userId') or sub.get('assignedStudent') or "unknown"
        surname = get_user_surname(cls, user_id)
        assignment = sub.get('assignmentSubmission', {})
        attachments = assignment.get('attachments', []) if assignment else []
        agg_text = ""
        for att in attachments:
            drive_att = att.get('driveFile') or {}
            drive_file = drive_att.get('driveFile') or drive_att  # compat dei diversi formati
            file_id = drive_file.get('id') if isinstance(drive_file, dict) else None
            if not file_id:
                continue
            txt = drive_file_to_text(drv, file_id)
            if txt:
                agg_text += "\n" + txt
        if agg_text.strip():
            texts.append(agg_text)
            # Nome con cognome, eventualmente aggiungi submissionId per univocità
            fname = f"{surname}_{sub.get('id', '')}.txt" if surname else f"student_{sub.get('id','')}.txt"
            names.append(fname)
    return texts, names

# --------------------- Funzioni esistenti ---------------------

def get_sentences(text):
    sentences = tokenize.sent_tokenize(text)
    return sentences

def get_url(sentence):
    base_url = 'https://www.google.com/search?q='
    query = sentence
    query = query.replace(' ', '+')
    url = base_url + query
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    divs = soup.find_all('div', class_='yuRUbf')
    urls = []
    for div in divs:
        a = div.find('a')
        urls.append(a['href'])
    if len(urls) == 0:
        return None
    elif "youtube" in urls[0]:
        return None
    else:
        return urls[0]

def read_text_file(file):
    content = ""
    with io.open(file.name, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def read_docx_file(file):
    text = docx2txt.process(file)
    return text

def read_pdf_file(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_from_file(uploaded_file):
    content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            content = read_text_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            content = read_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = read_docx_file(uploaded_file)
        if uploaded_file.type == "application/octet-stream" and uploaded_file.name.endswith(".java"):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            content = stringio.read()
    return content

def get_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

def get_similarity(text1, text2):
    text_list = [text1, text2]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    similarity = cosine_similarity(count_matrix)[0][1]
    return similarity

def get_similarity_list(texts, filenames=None):
    similarity_list = []
    if filenames is None:
        filenames = [f"File {i+1}" for i in range(len(texts))]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = get_similarity(texts[i], texts[j])
            similarity_list.append((filenames[i], filenames[j], similarity))
    return similarity_list
def get_similarity_list2(text, url_list):
    similarity_list = []
    for url in url_list:
        text2 = get_text(url)
        similarity = get_similarity(text, text2)
        similarity_list.append(similarity)
    return similarity_list

def plot_scatter(df):
    fig = px.scatter(df, x='File 1', y='File 2', color='Similarity', title='Similarity Scatter Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df):
    fig = px.line(df, x='File 1', y='File 2', color='Similarity', title='Similarity Line Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df):
    fig = px.bar(df, x='File 1', y='Similarity', color='File 2', title='Similarity Bar Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_pie(df):
    fig = px.pie(df, values='Similarity', names='File 1', title='Similarity Pie Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_box(df):
    fig = px.box(df, x='File 1', y='Similarity', title='Similarity Box Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(df):
    fig = px.histogram(df, x='Similarity', title='Similarity Histogram')
    st.plotly_chart(fig, use_container_width=True)

def plot_3d_scatter(df):
    fig = px.scatter_3d(df, x='File 1', y='File 2', z='Similarity', color='Similarity',
                        title='Similarity 3D Scatter Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_violin(df):
    fig = px.violin(df, y='Similarity', x='File 1', title='Similarity Violin Plot')
    st.plotly_chart(fig, use_container_width=True)



st.set_page_config(page_title='Plagiarism Detection')
st.title('Plagiarism Detector')

st.write("""
### Enter the text or upload a file to check for plagiarism or find similarities between files
""")
option = st.radio(
    "Select input option:",
    ('Enter text', 'Upload file', 'Find similarities between files', 'Google Classroom coursework')
)

if option == 'Enter text':
    text = st.text_area("Enter text here", height=200)
    uploaded_files = []
elif option == 'Upload file':
    uploaded_file = st.file_uploader("Upload file (." + ', .'.join(supported_file_types) + ")", type=supported_file_types)
    if uploaded_file is not None:
        text = get_text_from_file(uploaded_file)
        uploaded_files = [uploaded_file]
    else:
        text = ""
        uploaded_files = []
elif option == 'Find similarities between files':
    uploaded_files = st.file_uploader("Upload multiple files (." + ', .'.join(supported_file_types) + ")", type=supported_file_types, accept_multiple_files=True)
    texts = []
    filenames = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            text = get_text_from_file(uploaded_file)
            texts.append(text)
            filenames.append(uploaded_file.name)
    text = " ".join(texts)
else:
    # Google Classroom coursework
    st.subheader("Google Classroom: analizza consegne di un compito")
    client_secret_path = os.path.join(".streamlit", "google_client_secret.json")

    if not os.path.exists(client_secret_path):
        st.error("File .streamlit/google_client_secret.json non trovato. Aggiungilo al progetto prima di procedere.")
    else:
        # Mostra il redirect URI che verrà usato nel flow
        cs_redirect = build_codespaces_redirect_uri()
        if cs_redirect:
            st.info(f"Redirect URI che verrà usato per l'OAuth: {cs_redirect}. Assicurati che sia inserito tra gli 'Authorized redirect URIs' nelle credenziali OAuth di Google Cloud (tipo client Web).")
        params = dict(st.query_params)
        trigger_auth = st.button('Autentica con Google') or ('code' in params)
        if trigger_auth:
            try:
                creds = get_google_credentials(client_secret_path)
                st.success("Autenticazione completata.")
                st.session_state['google_authenticated'] = True
            except Exception as e:
                st.error(f"Autenticazione fallita: {e}")

    coursework_url = st.text_input("Incolla l'URL del compito Classroom (course_id e course_work_id)")

    if st.session_state.get('google_authenticated') and coursework_url:
        if st.button('Scarica consegne e analizza'):
            try:
                course_id, course_work_id = parse_coursework_url(coursework_url)
                creds = st.session_state.get('google_creds')
                with st.spinner('Recupero consegne e download allegati da Drive...'):
                    texts, filenames = extract_texts_from_submissions(creds, course_id, course_work_id)
                if not texts:
                    st.warning("Nessun testo trovato nelle consegne (nessun allegato o permessi insufficienti).")
                else:
                    # Pairwise similarities tra consegne
                    similarities = get_similarity_list(texts, filenames)
                    df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity'])
                    df = df.sort_values(by=['Similarity'], ascending=False)
                    st.write("Anteprima testi per studente (lunghezze):")
                    lengths = pd.DataFrame({
                        'File': filenames,
                        'Chars': [len(t or '') for t in texts]
                    }).sort_values('Chars', ascending=False)
                    st.dataframe(lengths, use_container_width=True)
                    # Grafici
                    plot_scatter(df)
                    plot_line(df)
                    plot_bar(df)
                    plot_pie(df)
                    plot_box(df)
                    plot_histogram(df)
                    plot_3d_scatter(df)
                    plot_violin(df)
            except Exception as e:
                st.error(f"Errore durante l'analisi Classroom: {e}")
    # Per questo flusso non usiamo la variabile 'text' né il pulsante generale sotto.
    text = ""

if st.button('Check for plagiarism or find similarities'):
    st.write("""
    ### Checking for plagiarism or finding similarities...
    """)
    if not text:
        st.write("""
        ### No text found for plagiarism check or finding similarities.
        """)
        st.stop()
    
    if option == 'Find similarities between files':
        similarities = get_similarity_list(texts, filenames)
        df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity'])
        df = df.sort_values(by=['Similarity'], ascending=False)
        # Plotting interactive graphs
        plot_scatter(df)
        plot_line(df)
        plot_bar(df)
        plot_pie(df)
        plot_box(df)
        plot_histogram(df)
        plot_3d_scatter(df)
        plot_violin(df)
    else:
        sentences = get_sentences(text)
        url = []
        for sentence in sentences:
            url.append(get_url(sentence))

        if None in url:
            st.write("""
            ### No plagiarism detected!
            """)
            st.stop()

        similarity_list = get_similarity_list2(text, url)
        df = pd.DataFrame({'Sentence': sentences, 'URL': url, 'Similarity': similarity_list})
        df = df.sort_values(by=['Similarity'], ascending=True)
    
    df = df.reset_index(drop=True)
    
    # Make URLs clickable in the DataFrame
    if 'URL' in df.columns:
        df['URL'] = df['URL'].apply(lambda x: '<a href="{}">{}</a>'.format(x, x) if x else '')
    
    # Center align URL column header
    df_html = df.to_html(escape=False)
    if 'URL' in df.columns:
        df_html = df_html.replace('<th>URL</th>', '<th style="text-align: center;">URL</th>')
    st.write(df_html, unsafe_allow_html=True)

