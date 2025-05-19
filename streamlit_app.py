import streamlit as st
import openai
import os
import tempfile
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "prompt.txt"
ESEMPI_DIR = BASE_DIR / "esempi"
NUM_ESEMPI = 6 # Number of zanzaraX.txt files

# --- Helper Functions to Load External Files ---
@st.cache_data # Cache to avoid re-reading files on every interaction
def load_main_prompt():
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
    esempi_content = []
    for i in range(1, NUM_ESEMPI + 1):
        file_path = ESEMPI_DIR / f"zanzara{i}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                esempi_content.append(f.read().strip())
        except FileNotFoundError:
            st.error(f"Errore: Il file '{file_path.name}' non è stato trovato in '{ESEMPI_DIR.name}'.")
            # Continue loading other examples if possible, or st.stop()
            continue # Or st.stop() if all examples are critical
        except Exception as e:
            st.error(f"Errore durante la lettura di '{file_path.name}': {e}")
            continue
    if len(esempi_content) != NUM_ESEMPI:
        st.warning(f"Attenzione: Caricati {len(esempi_content)} esempi su {NUM_ESEMPI} previsti dalla directory '{ESEMPI_DIR.name}'.")
    return esempi_content

# --- OpenAI API Key Handling ---
try:
    # Attempt to get the API key from Streamlit secrets first
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
except (AttributeError, KeyError): # Handle cases where st.secrets might not be available or key not set
    openai_api_key = None


st.set_page_config(layout="wide")
st.title("🎙️ Zanzara Tigre generator 📰")

if not openai_api_key:
    openai_api_key_input = st.text_input(
        "🔑 Inserisci la tua OpenAI API Key (o configurala nei secrets di Streamlit Cloud):",
        type="password",
        key="api_key_input_main" # Add a key for uniqueness
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
uploaded_file = st.file_uploader(
    "Scegli un file audio (es. MP3, WAV, M4A) o video (es. MP4, MOV, AVI)",
    type=['mp3', 'wav', 'm4a', 'mp4', 'mov', 'avi', 'ogg', 'webm']
)

# --- Optional User Instructions ---
st.header("2. (Opzionale) Aggiungi istruzioni personalizzate")
user_additional_instructions = st.text_area(
    "Se hai istruzioni specifiche aggiuntive per GPT-4o, inseriscile qui. Verranno aggiunte al prompt.",
    height=100,
    placeholder="Es: Enfatizza gli aspetti controversi, oppure adotta un tono più formale..."
)


def gpt_request(model_id, query_content, temperature=0.8):
    """
    Sends a request to the OpenAI GPT model.
    """
    messages = [{'role': 'user', 'content': query_content}]
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        return response # Return the full response object
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
    st.audio(uploaded_file, format=uploaded_file.type if uploaded_file.type else None)

    st.header("3. Avvia l'analisi")
    if st.button("🚀 Trascrivi e Analizza con GPT-4o"):
        # Load prompt components
        main_prompt_instruction = load_main_prompt()
        esempi_texts = load_esempi()

        if not main_prompt_instruction or not esempi_texts: # Basic check, more robust checks in loading functions
            st.error("Impossibile caricare i componenti del prompt dai file. Controlla i file e i log.")
            st.stop()

        with st.spinner("Attendere prego: Trascrizione audio in corso..."):
            trascrizione = "" # Initialize to ensure it's defined
            try:
                # Save the uploaded file temporarily to pass its path to OpenAI
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Call Whisper API for transcription
                with open(tmp_file_path, "rb") as audio_file_to_transcribe:
                    transcript_response = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=audio_file_to_transcribe,
                        response_format="text" # Get plain text directly
                    )
                trascrizione = transcript_response.text.replace("\n", " ").replace("  ", " ")

                # Clean up the temporary file
                os.remove(tmp_file_path)

                st.subheader("📄 Trascrizione Ottenuta:")
                with st.expander("Mostra/Nascondi Trascrizione", expanded=False):
                    st.text_area("Testo trascritto:", trascrizione, height=150, key="transcription_output")

            except openai.APIConnectionError as e:
                st.error(f"Errore di connessione API OpenAI (Whisper): {e}")
                st.stop()
            except openai.RateLimitError as e:
                st.error(f"Rate limit superato per l'API OpenAI (Whisper): {e}")
                st.stop()
            except openai.APIStatusError as e:
                st.error(f"Errore API OpenAI (Whisper): {e.status_code} - {e.response}")
                st.stop()
            except Exception as e:
                st.error(f"Errore durante la trascrizione: {e}")
                st.stop()

        if trascrizione: # Proceed only if transcription was successful
            with st.spinner("Attendere prego: Analisi GPT-4o in corso..."):
                # Construct the <esempi> block
                esempi_block = "<esempi>\n"
                for esempio_text in esempi_texts:
                    esempi_block += f"        <esempio>\n            {esempio_text}\n        </esempio>\n"
                esempi_block += "</esempi>"

                # Prepare user's additional instructions
                additional_instructions_block = ""
                if user_additional_instructions and user_additional_instructions.strip():
                    additional_instructions_block = f"\n<additional_instructions>\n{user_additional_instructions.strip()}\n</additional_instructions>\n"

                # Construct the full prompt for GPT-4o
                full_prompt_for_gpt = f"""{main_prompt_instruction}

{esempi_block}
{additional_instructions_block}
<input>
    {trascrizione}
</input>
"""
                if st.checkbox("Mostra prompt completo inviato a GPT-4o (per debug)", False, key="show_prompt_checkbox"):
                    st.text_area("Prompt GPT-4o:", full_prompt_for_gpt, height=300, key="full_prompt_display")

                # Call GPT-4o
                gpt_response_obj = gpt_request("gpt-4o", full_prompt_for_gpt, 0.8)

                if gpt_response_obj:
                    risultato_gpt = gpt_response_obj.choices[0].message.content
                    st.subheader("📰 Risultato dall'Analisi GPT-4o:")
                    st.markdown(risultato_gpt)
                else:
                    st.error("Non è stato possibile ottenere una risposta da GPT-4o.")
        else:
            st.error("La trascrizione è vuota o non è stata completata. Impossibile procedere con l'analisi GPT.")
else:
    st.info("Carica un file audio o video per iniziare.")

st.markdown("---")
st.markdown("App creata con Streamlit e OpenAI. Legge il prompt da `prompt.txt` e gli esempi dalla directory `esempi/`.")