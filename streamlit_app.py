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
GPT_MODEL = "gpt-4.1"  # Added constant for GPT model
TRANSCRIPTION_MODEL = "gpt-4o-mini-audio-preview"  # Added constant for Transcription model

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
            continue
        except Exception as e:
            st.error(f"Errore durante la lettura di '{file_path.name}': {e}")
            continue

    if not esempi_content:
        st.warning(
            f"Attenzione: Nessun file di esempio (zanzaraX.txt) è stato trovato nella directory '{ESEMPI_DIR.name}'. "
            f"L'analisi {GPT_MODEL} procederà senza esempi specifici, il che potrebbe influire sulla qualità del risultato."
        )
    elif len(esempi_content) != NUM_ESEMPI:
        st.warning(f"Attenzione: Caricati {len(esempi_content)} esempi su {NUM_ESEMPI} previsti dalla directory '{ESEMPI_DIR.name}'.")
    return esempi_content

# --- OpenAI API Key Handling ---
try:
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
except (AttributeError, KeyError):
    openai_api_key = None

st.set_page_config(layout="wide")
st.title(f"🎙️ Zanzara Tigre generator ({GPT_MODEL} & {TRANSCRIPTION_MODEL}) 📰") # Updated title

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
supported_formats = ['flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm']
uploaded_file = st.file_uploader(
    f"Scegli un file audio o video (formati supportati: {', '.join(supported_formats)})",
    type=supported_formats
)

# --- Optional User Instructions ---
st.header("2. Opzioni di Elaborazione")
user_additional_instructions = st.text_area(
    f"Se hai istruzioni specifiche aggiuntive per {GPT_MODEL} (verranno ignorate se si sceglie solo trascrizione), inseriscile qui. Verranno aggiunte al prompt.",
    height=100,
    placeholder="Es: Enfatizza gli aspetti controversi, oppure adotta un tono più formale..."
)

# --- Transcription-only Flag ---
transcribe_only = st.checkbox(
    f"✅ Genera solo la trascrizione (salta l'analisi {GPT_MODEL})",
    value=False,
    key="transcribe_only_checkbox",
    help=f"Se selezionato, verrà generata solo la trascrizione del file audio/video con {TRANSCRIPTION_MODEL}, senza inviarla a {GPT_MODEL}."
)

def gpt_request(model_id, query_content, temperature=0.8): # model_id is already a parameter
    """Sends a request to the OpenAI Chat Completions API."""
    messages = [{'role': 'user', 'content': query_content}]
    try:
        response = client.chat.completions.create(
            model=model_id, # Uses the passed model_id
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
    st.audio(uploaded_file, format=uploaded_file.type if uploaded_file.type else None)

    st.header("3. Avvia Elaborazione")

    if transcribe_only:
        button_label = f"🎤 Trascrivi Solamente ({TRANSCRIPTION_MODEL})"
    else:
        button_label = f"🚀 Trascrivi ({TRANSCRIPTION_MODEL}) e Analizza con {GPT_MODEL}"

    if st.button(button_label, key="process_button"):
        main_prompt_instruction = load_main_prompt()
        esempi_texts = load_esempi()

        if not main_prompt_instruction and not transcribe_only:
            st.error(f"Impossibile caricare il prompt principale ('prompt.txt'). Necessario per l'analisi {GPT_MODEL}.")
            st.stop()

        MAX_BYTES = 25 * 1024 * 1024
        if uploaded_file.size > MAX_BYTES:
            st.error(f"Il file caricato ({uploaded_file.size / (1024*1024):.2f} MB) supera il limite di 25 MB per l'API {TRANSCRIPTION_MODEL}. Riduci la dimensione del file o usa un estratto più breve.")
            st.stop()

        st.write("Dettagli file caricato:", {
            "Nome": uploaded_file.name,
            "Dimensione (bytes)": uploaded_file.size,
            "Tipo": uploaded_file.type
        })

        trascrizione = None
        try:
            with st.spinner(f"Attendere prego: Trascrizione audio con {TRANSCRIPTION_MODEL} in corso..."):
                uploaded_file.seek(0)
                transcript_response = client.audio.transcriptions.create(
                    model=TRANSCRIPTION_MODEL, # Use constant
                    file=uploaded_file,
                    response_format="text"
                )
            
            if isinstance(transcript_response, str):
                 trascrizione = transcript_response.replace("\n", " ").strip()
            else:
                 trascrizione = str(transcript_response).replace("\n", " ").strip()

            st.subheader(f"📄 Trascrizione Ottenuta (da {TRANSCRIPTION_MODEL}):")
            st.text_area(
                "Testo trascritto (puoi selezionare e copiare questo testo):",
                value=trascrizione if trascrizione else "Trascrizione non disponibile o vuota.",
                height=250,
                key="transcription_output_area",
                help="Clicca all'interno di quest'area e usa Ctrl+C (o Cmd+C su Mac) per copiare il testo."
            )
        except Exception as e:
            st.error(f"Errore durante la trascrizione {TRANSCRIPTION_MODEL}: {e}")
            st.exception(e)
            st.stop()

        if trascrizione:
            if not transcribe_only:
                if not main_prompt_instruction:
                    st.error(f"Prompt principale mancante, impossibile procedere con l'analisi {GPT_MODEL}.")
                    st.stop()

                with st.spinner(f"Attendere prego: Analisi {GPT_MODEL} in corso..."):
                    esempi_block = "<esempi>\n"
                    if esempi_texts:
                        for esempio_text in esempi_texts:
                            esempi_block += f"        <esempio>\n            {esempio_text}\n        </esempio>\n"
                    else:
                        esempi_block += "        \n"
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

                    if st.checkbox(f"Mostra prompt completo inviato a {GPT_MODEL} (per debug)", False, key="show_prompt_checkbox"):
                        st.text_area(f"Prompt {GPT_MODEL}:", full_prompt_for_gpt, height=300, key="full_prompt_display")

                    gpt_response_obj = gpt_request(GPT_MODEL, full_prompt_for_gpt, 0.8) # Use constant

                    if gpt_response_obj and gpt_response_obj.choices:
                        risultato_gpt = gpt_response_obj.choices[0].message.content.strip()
                        st.subheader(f"📰 Risultato dall'Analisi {GPT_MODEL}:")
                        st.markdown(risultato_gpt)

                        with st.expander("Copia il risultato dell'analisi (testo grezzo)", expanded=True):
                            st.text_area(
                                "Testo dell'analisi (per copia-incolla):",
                                value=risultato_gpt,
                                height=300,
                                key="gpt_result_copy_area",
                                help="Clicca all'interno di quest'area e usa Ctrl+C (o Cmd+C su Mac) per copiare il testo."
                            )
                    else:
                        st.error(f"Non è stato possibile ottenere una risposta da {GPT_MODEL} o la risposta era vuota.")
            elif trascrizione:
                st.success(f"✔️ Trascrizione con {TRANSCRIPTION_MODEL} completata. L'analisi {GPT_MODEL} è stata saltata come richiesto.")
            
            if not trascrizione.strip() and not transcribe_only:
                st.warning(f"La trascrizione è risultata vuota. L'analisi {GPT_MODEL} è stata eseguita su testo vuoto (se non saltata).")
            elif not trascrizione.strip() and transcribe_only:
                 st.warning("La trascrizione è risultata vuota.")
else:
    st.info("Carica un file audio o video per iniziare.")

st.markdown("---")
st.markdown(f"App creata con Streamlit e OpenAI. Modelli usati: {TRANSCRIPTION_MODEL} per la trascrizione, {GPT_MODEL} per l'analisi. Legge il prompt da `prompt.txt` e gli esempi (opzionali) dalla directory `esempi/`.")