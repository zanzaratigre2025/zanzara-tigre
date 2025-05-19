import streamlit as st
import openai
import os
from pathlib import Path
from io import BytesIO

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "prompt.txt"
ESEMPI_DIR = BASE_DIR / "esempi"
NUM_ESEMPI = 6  # Number of zanzaraX.txt files

# --- Helper Functions to Load External Files ---
@st.cache_data
def load_main_prompt():
    """Loads the main prompt from prompt.txt."""
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error(f"Errore: Il file '{PROMPT_FILE.name}' non è stato trovato nella directory principale.")
        st.stop()
    except Exception as e:
        st.error(f"Errore durante la lettura di '{PROMPT_FILE.name}': {e}")
        st.stop()

@st.cache_data
def load_esempi():
    """Loads example texts from the esempi/ directory."""
    esempi_content = []
    for i in range(1, NUM_ESEMPI + 1):
        file_path = ESEMPI_DIR / f"zanzara{i}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                esempi_content.append(f.read().strip())
        except FileNotFoundError:
            st.error(f"Errore: Il file '{file_path.name}' non è stato trovato in '{ESEMPI_DIR.name}'.")
            # Continue to load other examples if one is missing
            continue
        except Exception as e:
            st.error(f"Errore durante la lettura di '{file_path.name}': {e}")
            continue

    if len(esempi_content) != NUM_ESEMPI:
        st.warning(f"Attenzione: Caricati {len(esempi_content)} esempi su {NUM_ESEMPI} previsti dalla directory '{ESEMPI_DIR.name}'.")
    return esempi_content

# --- OpenAI API Key Handling ---
try:
    # Attempt to get API key from Streamlit secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
except (AttributeError, KeyError):
    # Fallback if st.secrets is not available or key is not found
    openai_api_key = None

st.set_page_config(layout="wide")
st.title("🎙️ Zanzara Tigre generator 📰")

if not openai_api_key:
    openai_api_key_input = st.text_input(
        "🔑 Inserisci la tua OpenAI API Key (o configurala nei secrets di Streamlit Cloud):",
        type="password",
        key="api_key_input_main",
        help="La tua API key è necessaria per utilizzare i servizi OpenAI."
    )
    if openai_api_key_input:
        openai_api_key = openai_api_key_input
    else:
        st.warning("Per favore inserisci la tua OpenAI API Key per continuare.")
        st.stop()
else:
    st.success("OpenAI API Key caricata correttamente dai secrets. ✅")

# Initialize OpenAI client
try:
    client = openai.OpenAI(api_key=openai_api_key)
except Exception as e:
    st.error(f"Errore nell'inizializzazione del client OpenAI: {e}")
    st.stop()

# --- File Uploader ---
st.header("1. Carica il tuo file audio o video")
# Supported formats by OpenAI Whisper API
# https://platform.openai.com/docs/guides/speech-to-text/supported-formats
supported_formats = ['flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm']
uploaded_file = st.file_uploader(
    f"Scegli un file audio o video (formati supportati: {', '.join(supported_formats)})",
    type=supported_formats
)

# --- Optional User Instructions ---
st.header("2. (Opzionale) Aggiungi istruzioni personalizzate")
user_additional_instructions = st.text_area(
    "Se hai istruzioni specifiche aggiuntive per GPT-4o, inseriscile qui. Verranno aggiunte al prompt.",
    height=100,
    placeholder="Es: Enfatizza gli aspetti controversi, oppure adotta un tono più formale..."
)

def gpt_request(model_id, query_content, temperature=0.8):
    """Sends a request to the OpenAI Chat Completions API."""
    messages = [{'role': 'user', 'content': query_content}]
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        return response
    except openai.APIConnectionError as e:
        st.error(f"Errore di connessione API OpenAI: {e}")
    except openai.RateLimitError as e:
        st.error(f"Rate limit superato per l'API OpenAI: {e}")
    except openai.APIStatusError as e:
        st.error(f"Errore API OpenAI: {e.status_code} - {e.response}")
    except Exception as e:
        st.error(f"Un errore inaspettato è occorso durante la richiesta GPT: {e}")
    return None

if uploaded_file is not None:
    # Display the uploaded audio file
    st.audio(uploaded_file, format=uploaded_file.type if uploaded_file.type else None)

    st.header("3. Avvia l'analisi")
    if st.button("🚀 Trascrivi e Analizza con GPT-4o"):
        main_prompt_instruction = load_main_prompt()
        esempi_texts = load_esempi()

        if not main_prompt_instruction: # Examples can be empty if not all found
            st.error("Impossibile caricare il prompt principale. Controlla il file 'prompt.txt' e i log.")
            st.stop()

        # --- File size check ---
        # OpenAI Whisper API has a 25 MB file size limit.
        MAX_BYTES = 25 * 1024 * 1024  # 25 MB
        if uploaded_file.size > MAX_BYTES:
            st.error(f"Il file caricato ({uploaded_file.size / (1024*1024):.2f} MB) supera il limite di 25 MB per l'API Whisper. Riduci la dimensione del file o usa un estratto più breve.")
            st.stop()

        # --- Debug info ---
        st.write({
            "uploaded_file_name": uploaded_file.name,
            "uploaded_file_size_bytes": uploaded_file.size,
            "uploaded_file_type": uploaded_file.type
        })

        trascrizione = None
        try:
            with st.spinner("Attendere prego: Trascrizione audio in corso..."):
                # Reset file pointer before sending to API
                uploaded_file.seek(0)
                # Transcribe using Whisper API
                # The `uploaded_file` object (a BytesIO subclass) can be passed directly.
                # The OpenAI SDK will handle reading its name and content type.
                transcript_obj = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=uploaded_file, # Pass the UploadedFile object directly
                    response_format="text" # Get plain text output
                )
            
            # `transcript_obj` is a string when response_format="text"
            # For newer versions of openai client, it might be an object with a .text attribute
            if isinstance(transcript_obj, str):
                 trascrizione = transcript_obj.replace("\n", " ").strip()
            else:
                 # Assuming it's a Transcription object if not a string
                 trascrizione = transcript_obj.text.replace("\n", " ").strip()


            st.subheader("📄 Trascrizione Ottenuta:")
            with st.expander("Mostra/Nascondi Trascrizione", expanded=False):
                st.text_area("Testo trascritto:", trascrizione, height=150, key="transcription_output")
        except Exception as e:
            st.error(f"Errore durante la trascrizione Whisper: {e}")
            st.exception(e) # Provides more details for debugging
            st.stop()

        # --- GPT Analysis ---
        if trascrizione:
            with st.spinner("Attendere prego: Analisi GPT-4o in corso..."):
                esempi_block = "<esempi>\n"
                if esempi_texts: # Only add block if examples were loaded
                    for esempio_text in esempi_texts:
                        esempi_block += f"        <esempio>\n            {esempio_text}\n        </esempio>\n"
                esempi_block += "</esempi>"

                additional_instructions_block = ""
                if user_additional_instructions.strip():
                    additional_instructions_block = f"\n<additional_instructions>\n{user_additional_instructions.strip()}\n</additional_instructions>\n"

                full_prompt_for_gpt = f"""{main_prompt_instruction}

{esempi_block}
{additional_instructions_block}
<input>
    {trascrizione}
</input>
"""

                if st.checkbox("Mostra prompt completo inviato a GPT-4o (per debug)", False, key="show_prompt_checkbox"):
                    st.text_area("Prompt GPT-4o:", full_prompt_for_gpt, height=300, key="full_prompt_display")

                gpt_response_obj = gpt_request("gpt-4o", full_prompt_for_gpt, 0.8)

                if gpt_response_obj and gpt_response_obj.choices:
                    risultato_gpt = gpt_response_obj.choices[0].message.content
                    st.subheader("📰 Risultato dall'Analisi GPT-4o:")
                    st.markdown(risultato_gpt)
                else:
                    st.error("Non è stato possibile ottenere una risposta da GPT-4o o la risposta era vuota.")
        elif trascrizione is not None: # Check if transcription is empty string
             st.warning("La trascrizione è vuota. Impossibile procedere con l'analisi GPT.")
        else: # Transcription failed (trascrizione is None)
            st.error("La trascrizione non è stata completata. Impossibile procedere con l'analisi GPT.")
else:
    st.info("Carica un file audio o video per iniziare.")

st.markdown("---")
st.markdown("App creata con Streamlit e OpenAI. Legge il prompt da `prompt.txt` e gli esempi dalla directory `esempi/`.")
