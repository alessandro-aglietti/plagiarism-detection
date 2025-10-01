rm -Rf .venv ntlk_data .streamlit/token.json
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py