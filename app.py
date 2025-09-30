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
import base64
import google.generativeai as genai

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

# Utility salvataggio su FS
def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name or "file")
    return name[:120]

def _ensure_output_dir(course_id: str, course_work_id: str) -> str:
    out_dir = os.path.join(".streamlit", "downloads", str(course_id), str(course_work_id))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _make_download_link(path: str, label: str, filename: str | None = None, mime: str = 'text/plain') -> str:
    try:
        with open(path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode('utf-8')
        if not filename:
            filename = os.path.basename(path)
        href = f"data:{mime};base64,{b64}"
        return f'<a download="{filename}" href="{href}" target="_blank">{label}</a>'
    except Exception:
        return f'<span style="color:red;">Impossibile preparare il download</span>'

# Lettura API key di Gemini da file

def _load_gemini_api_key(path: str = os.path.join('.streamlit', 'gemini-api-key.txt')) -> str:
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        pass
    return ""

# ---------------- Gemini: valutazione migliore consegna ----------------

def list_gemini_models(api_key: str) -> List[str]:
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        names = []
        for m in models:
            methods = getattr(m, 'supported_generation_methods', []) or []
            if 'generateContent' in methods:
                # usa il nome completo così come restituito (es. models/gemini-1.5-flash-001)
                names.append(m.name)
        # ordina con preferenza a flash/pro
        names.sort()
        return names
    except Exception:
        return []

def analyze_best_submission_with_gemini(api_key: str, model_name: str, coursework_title: str, coursework_description: str, filenames: List[str], filepaths: List[str]) -> dict:
    if not api_key:
        raise ValueError("API key di Gemini mancante. Imposta la variabile d'ambiente GOOGLE_API_KEY o inseriscila nel campo dedicato.")
    if not model_name:
        raise ValueError("Modello Gemini non selezionato.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Carica i file (testo) e prepara la mappa nome->file
    uploads = []
    file_map = []
    for name, path in zip(filenames, filepaths):
        try:
            f = genai.upload_file(path=path, mime_type='text/plain')
            uploads.append(f)
            file_map.append({"filename": name, "uri": f.name})
        except Exception:
            # Se upload fallisce, prova a includere solo il testo in prompt
            uploads.append(None)
            file_map.append({"filename": name, "uri": None})

    prompt = (
        "Sei un revisore. Ti fornisco gli elaborati degli studenti per il coursework indicato. "
        "Valuta qualità complessiva, chiarezza, correttezza, completezza e originalità, scegli la migliore consegna e individua quelle incomplete rispetto ai requisiti. "
        "Rispondi in JSON con le chiavi: "
        "best_file (string, uno dei nomi file elencati), best_reason (string sintetica), "
        "worst_file (string), worst_reason (string sintetica), "
        "incomplete (array di oggetti {file: string, reason: string}).\n\n"
        f"Titolo coursework: [{coursework_title}]\n"
        f"Descrizione coursework: [{coursework_description}]\n\n"
        "Elenco file consegnati (nome -> risorsa):\n"
        + "\n".join([f"- {m['filename']} -> {m['uri'] or 'inline'}" for m in file_map])
        + "\n\nAnalizza attentamente gli elaborati e fornisci la scelta e le valutazioni."
    )

    contents = []
    # Inserisci prima le risorse file disponibili
    for f in uploads:
        if f is not None:
            contents.append(f)
    contents.append(prompt)

    resp = model.generate_content(contents)
    text = resp.text or ""
    # Prova a estrarre JSON
    result = {"raw": text}
    try:
        import json as _json
        # Trova primo blocco JSON nel testo
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            parsed = _json.loads(text[start:end+1])
            if isinstance(parsed, dict) and 'best_file' in parsed:
                result.update(parsed)
    except Exception:
        pass
    return result

# Scopes: Classroom lettura compiti/consegne e Drive read-only
def _load_scopes_from_file(path: str = os.path.join(".streamlit", "scopes.txt")) -> List[str]:
    scopes: List[str] = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Accetta separatori: newline, spazi, tab, virgole
            parts = re.split(r"[\s,]+", content.strip())
            scopes = [p for p in parts if p]
    except Exception:
        scopes = []
    return scopes

GOOGLE_SCOPES = _load_scopes_from_file()

# This disables the requested scopes and granted scopes check.
# If users only grant partial request, the warning would not be thrown.
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

# --------------------- Helper Google OAuth & Classroom/Drive ---------------------
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
                returned_scopes = params.get('scope')
                if isinstance(returned_scopes, list):
                    returned_scopes = returned_scopes[0]
                # scopes_list = returned_scopes.split(' ') if returned_scopes else GOOGLE_SCOPES
                scopes_list = GOOGLE_SCOPES
                flow = Flow.from_client_secrets_file(
                    client_secrets_file=client_secret_path,
                    scopes=scopes_list,
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
        st.markdown(f"<a href=\"{auth_url}\" target=\"_self\">Continua l'autenticazione Google</a>", unsafe_allow_html=True)
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


def classroom_service(creds: Credentials):
    return build('classroom', 'v1', credentials=creds)

def drive_service(creds: Credentials):
    return build('drive', 'v3', credentials=creds)

# Nuove funzioni: elenco corsi attivi e coursework

def list_active_courses(classroom) -> List[dict]:
    courses = []
    page_token = None
    while True:
        resp = classroom.courses().list(pageToken=page_token, courseStates=['ACTIVE']).execute()
        courses.extend(resp.get('courses', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return courses

def list_coursework_for_course(classroom, course_id: str) -> List[dict]:
    items = []
    page_token = None
    while True:
        resp = classroom.courses().courseWork().list(courseId=course_id, pageToken=page_token).execute()
        items.extend(resp.get('courseWork', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return items

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

def extract_texts_from_submissions(creds: Credentials, course_id: str, course_work_id: str) -> Tuple[List[str], List[str], List[str]]:
    """Per ogni consegna, concatena il testo degli allegati driveFile, salva un .txt su FS e usa il cognome come nome file.
    Ritorna (texts, filenames, filepaths).
    """
    cls = classroom_service(creds)
    drv = drive_service(creds)
    submissions = list_student_submissions(cls, course_id, course_work_id)
    texts: List[str] = []
    names: List[str] = []
    paths: List[str] = []
    out_dir = _ensure_output_dir(course_id, course_work_id)

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
            # Salva su FS
            base_name = _safe_filename(f"{surname}_{sub.get('id','')}.txt")
            file_path = os.path.join(out_dir, base_name)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(agg_text)
            except Exception:
                # fallback latin-1
                with open(file_path, "w", encoding="latin-1", errors="ignore") as f:
                    f.write(agg_text)
            texts.append(agg_text)
            names.append(base_name)
            paths.append(file_path)
    return texts, names, paths

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

    # Se autenticato, mostra selezione corso e coursework
    selected_course = None
    selected_coursework = None
    if st.session_state.get('google_authenticated'):
        try:
            creds = st.session_state.get('google_creds')
            cls = classroom_service(creds)
            courses = list_active_courses(cls)
            if not courses:
                st.warning("Nessun corso ACTIVE trovato per questo account.")
            else:
                selected_course = st.selectbox(
                    "Seleziona un corso",
                    options=courses,
                    format_func=lambda c: f"{c.get('name','(senza nome)')} [{c.get('id')}]"
                )
                if selected_course:
                    courseworks = list_coursework_for_course(cls, selected_course.get('id'))
                    if not courseworks:
                        st.info("Nessun coursework trovato per questo corso.")
                    else:
                        selected_coursework = st.selectbox(
                            "Seleziona un coursework",
                            options=courseworks,
                            format_func=lambda w: f"{w.get('title','(senza titolo)')} [{w.get('id')}]"
                        )
        except Exception as e:
            st.error(f"Errore durante il caricamento di corsi/coursework: {e}")

    # Avvio analisi se corso e compito selezionati
    if selected_course and selected_coursework:
        if st.button('Scarica consegne e analizza'):
            try:
                course_id = selected_course.get('id')
                course_work_id = selected_coursework.get('id')
                creds = st.session_state.get('google_creds')
                with st.spinner('Recupero consegne e download allegati da Drive...'):
                    texts, filenames, filepaths = extract_texts_from_submissions(creds, course_id, course_work_id)
                if not texts:
                    st.warning("Nessun testo trovato nelle consegne (nessun allegato o permessi insufficienti).")
                else:
                    # Calcola similarità e salva tutto in sessione per persistere tra i rerun
                    similarities = get_similarity_list(texts, filenames)
                    st.session_state['classroom_analysis'] = {
                        'course_id': course_id,
                        'course_work_id': course_work_id,
                        'coursework_title': selected_coursework.get('title', '(senza titolo)'),
                        'coursework_description': selected_coursework.get('description', ''),
                        'texts': texts,
                        'filenames': filenames,
                        'filepaths': filepaths,
                        'similarities': similarities,
                    }
                    st.success("Consegne scaricate e analisi calcolata.")
            except Exception as e:
                st.error(f"Errore durante l'analisi Classroom: {e}")
    # Per questo flusso non usiamo la variabile 'text' né il pulsante generale sotto.
    text = ""

    # Visualizzazione risultati persistenti (grafici, download, Gemini)
    analysis = st.session_state.get('classroom_analysis')
    if analysis:
        filenames = analysis.get('filenames', [])
        filepaths = analysis.get('filepaths', [])
        texts = analysis.get('texts', [])
        similarities = analysis.get('similarities', [])
        course_work_title = analysis.get('coursework_title', '(senza titolo)')
        course_work_desc = analysis.get('coursework_description', '')
        if filenames and filepaths and texts and similarities:
            df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity'])
            df = df.sort_values(by=['Similarity'], ascending=False)
            st.write("Anteprima testi per studente (lunghezze):")
            lengths = pd.DataFrame({
                'File': filenames,
                'Path': filepaths,
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

            # Proponi download dei file più simili (top 5 coppie)
            st.subheader("Scarica le coppie di elaborati più simili")
            path_map = {os.path.basename(p): p for p in filepaths}
            top_pairs = df.head(5).to_dict(orient='records')
            for i, row in enumerate(top_pairs, start=1):
                f1 = row['File 1']
                f2 = row['File 2']
                sim = row['Similarity']
                p1 = path_map.get(f1)
                p2 = path_map.get(f2)
                cols = st.columns(3)
                cols[0].markdown(f"**Coppia {i}** — Similarità: {sim:.3f}")
                if p1 and os.path.exists(p1):
                    cols[1].markdown(_make_download_link(p1, f"Scarica {f1}", f1), unsafe_allow_html=True)
                else:
                    cols[1].write("File 1 non disponibile")
                if p2 and os.path.exists(p2):
                    cols[2].markdown(_make_download_link(p2, f"Scarica {f2}", f2), unsafe_allow_html=True)
                else:
                    cols[2].write("File 2 non disponibile")

            # Sezione Gemini (persistente)
            st.subheader("Valutazione automatica con Gemini")
            default_key = _load_gemini_api_key()
            api_key_input = st.text_input(
                "Gemini API Key (lascia vuoto per usare la chiave nel file .streamlit/gemini-api-key.txt)",
                value=default_key,
                type='password',
                key='gemini_key_input'
            )
            # Carica e mostra i modelli disponibili (persisti in sessione per evitare richieste ripetute)
            key_used_for_models = st.session_state.get('gemini_models_key')
            models_list = st.session_state.get('gemini_models', [])
            current_key = api_key_input or _load_gemini_api_key()
            if current_key and (not models_list or key_used_for_models != current_key):
                models_list = list_gemini_models(current_key)
                st.session_state['gemini_models'] = models_list
                st.session_state['gemini_models_key'] = current_key
            if not models_list:
                st.info("Nessun modello disponibile. Inserisci una API key valida e aggiorna.")
            # Select modello
            def _model_label(name: str) -> str:
                try:
                    return name.split('/')[-1]
                except Exception:
                    return name
            selected_model = st.selectbox(
                "Seleziona il modello Gemini",
                options=models_list,
                format_func=_model_label,
                key='gemini_selected_model'
            )
            # Analisi con Gemini
            if st.button("Chiedi a Gemini la migliore consegna", key='gemini_eval_button'):
                try:
                    key_to_use = api_key_input or _load_gemini_api_key()
                    with st.spinner('Analisi con Gemini in corso...'):
                        gemini_result = analyze_best_submission_with_gemini(key_to_use, selected_model, course_work_title, course_work_desc, filenames, filepaths)
                    st.session_state['gemini_result'] = gemini_result
                except Exception as e:
                    st.session_state['gemini_result'] = {'error': str(e)}
            # Mostra risultato Gemini se presente
            gemres = st.session_state.get('gemini_result')
            if gemres:
                if 'error' in gemres:
                    st.error(f"Errore nell'analisi con Gemini: {gemres['error']}")
                elif 'best_file' in gemres:
                    st.success(f"Migliore consegna secondo Gemini: {gemres['best_file']}")
                    st.write(gemres.get('reasoning', ''))
                    # Tabella di valutazione finale: cognome, valutazione, motivazione sintetica
                    rows = []
                    # Helper per cognome da filename (prefisso prima di _)
                    def _surname_from_filename(fn: str) -> str:
                        try:
                            return os.path.basename(fn).split('_')[0]
                        except Exception:
                            return fn
                    best = gemres.get('best_file')
                    worst = gemres.get('worst_file')
                    best_reason = gemres.get('best_reason', '')
                    worst_reason = gemres.get('worst_reason', '')
                    incomplete_list = gemres.get('incomplete', []) or []
                    # Mappa filename->(valutazione, motivazione)
                    eval_map = {}
                    if isinstance(incomplete_list, list):
                        for it in incomplete_list:
                            try:
                                f = it.get('file') if isinstance(it, dict) else None
                                r = it.get('reason') if isinstance(it, dict) else ''
                                if f:
                                    eval_map[f] = ('incompleto', r or '')
                            except Exception:
                                pass
                    if best:
                        eval_map[best] = ('migliore', best_reason)
                    if worst:
                        # Evita di sovrascrivere un best marcato anche come incompleto per errore del modello
                        if eval_map.get(worst, ('', ''))[0] != 'migliore':
                            eval_map[worst] = ('peggiore', worst_reason)
                    # Costruisci righe per tutti i file conosciuti
                    for fn in filenames:
                        label, reason = eval_map.get(fn, ('—', ''))
                        rows.append({
                            'Cognome': _surname_from_filename(fn),
                            'File': fn,
                            'Valutazione': label,
                            'Motivazione': reason
                        })
                    eval_df = pd.DataFrame(rows)
                    # Mostra prima migliore, poi incompleti, poi peggiore, poi altri
                    order = {'migliore': 0, 'incompleto': 1, 'peggiore': 2, '—': 3}
                    eval_df['order'] = eval_df['Valutazione'].map(lambda x: order.get(x, 4))
                    eval_df = eval_df.sort_values(['order', 'Cognome']).drop(columns=['order'])
                    st.dataframe(eval_df, use_container_width=True)
                 else:
                     st.info("Risposta di Gemini (non in formato JSON atteso):")
                     st.write(gemres.get('raw', ''))

