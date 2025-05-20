import streamlit as st
import openai
import os
from pathlib import Path
# from io import BytesIO # No longer strictly needed with current OpenAI SDK and Streamlit file handling

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "prompt.txt"
ESEMPI_DIR = BASE_DIR / "esempi"
NUM_ESEMPI = 6  # Number of zanzaraX.txt files
GPT_MODEL = "gpt-4o"  # GPT model for analysis
TRANSCRIPTION_MODEL = "whisper-1"  # Transcription model

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
            # Non-critical, so warning instead of error
            st.warning(f"Attenzione: Il file di esempio '{file_path.name}' non è stato trovato in '{ESEMPI_DIR.name}'.")
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
except (AttributeError, KeyError): # Handle cases where st.secrets might not be available or key missing
    openai_api_key = None

st.set_page_config(layout="wide")
st.title(f"🎙️ Zanzara Tigre generator ({GPT_MODEL} & {TRANSCRIPTION_MODEL}) 📰")

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
        st.error(f"Errore API OpenAI: Status {e.status_code} - {e.message}")
    except Exception as e:
        st.error(f"Un errore inaspettato è occorso durante la richiesta a {model_id}: {e}")
    return None

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type if uploaded_file.type else None)

    st.header("3. Avvia Elaborazione")

    if transcribe_only:
        button_label = f"🎤 Trascrivi Solamente ({TRANSCRIPTION_MODEL})"
    else:
        button_label = f"🚀 Trascrivi ({TRANSCRIPTION_MODEL}) e Analizza con {GPT_MODEL}"

    if st.button(button_label, key="process_button"):
        # File size check (OpenAI Whisper API limit is 25 MB)
        MAX_BYTES = 25 * 1024 * 1024
        if uploaded_file.size > MAX_BYTES:
            st.error(
                f"Il file caricato ({uploaded_file.size / (1024*1024):.2f} MB) supera il limite di 25 MB "
                f"per l'API {TRANSCRIPTION_MODEL}. Riduci la dimensione del file o usa un estratto più breve."
            )
            st.stop()

        st.write("Dettagli file caricato:", {
            "Nome": uploaded_file.name,
            "Dimensione (bytes)": uploaded_file.size,
            "Tipo": uploaded_file.type
        })

        # --- 1. Transcription ---
        trascrizione_raw = None
        with st.spinner(f"Attendere prego: Trascrizione audio con {TRANSCRIPTION_MODEL} in corso..."):
            try:
                uploaded_file.seek(0) # Reset file pointer to the beginning
                # The OpenAI client expects the file object directly.
                # Streamlit's UploadedFile is a BytesIO-like object, which is compatible.
                transcript_response_text = client.audio.transcriptions.create(
                    model=TRANSCRIPTION_MODEL,
                    file=uploaded_file, # Pass the file object directly
                    response_format="text" # Ensures the response is plain text
                )
                # For response_format="text", the SDK directly returns the transcribed string.
                trascrizione_raw = transcript_response_text.strip()

            except openai.APIError as e: # Catch specific OpenAI errors
                st.error(f"Errore API OpenAI durante la trascrizione con {TRANSCRIPTION_MODEL}: {e}")
                st.exception(e) # Provides full traceback for debugging in logs/console
                st.stop()
            except Exception as e: # Catch any other exceptions during transcription
                st.error(f"Errore generico durante la trascrizione con {TRANSCRIPTION_MODEL}: {e}")
                st.exception(e)
                st.stop()

        st.subheader(f"📄 Trascrizione Ottenuta (da {TRANSCRIPTION_MODEL}):")
        if trascrizione_raw: # Check if transcription is not empty
            st.code(trascrizione_raw, language='text')
            st.caption("Trascrizione completa. Usa l'icona di copia (solitamente in alto a destra del blocco di testo) per copiare.")
        else:
            # This handles cases where transcription might be an empty string (e.g., silent audio)
            # or if it somehow remained None (though st.stop() in except block should prevent this).
            st.warning("La trascrizione è risultata vuota.")
            if trascrizione_raw is None: # Should ideally not be reached if errors are caught
                st.error("Errore critico: la trascrizione non è disponibile. Elaborazione interrotta.")
                st.stop()


        # --- 2. Conditional GPT Analysis ---
        if transcribe_only:
            if trascrizione_raw: # Only show success if transcription has content
                st.success(f"✔️ Trascrizione con {TRANSCRIPTION_MODEL} completata. L'analisi {GPT_MODEL} è stata saltata come richiesto.")
            # If trascrizione_raw is empty, the warning above is sufficient.
        else:
            # Proceed with GPT analysis
            if not trascrizione_raw:
                st.warning(
                    f"La trascrizione è risultata vuota. L'analisi {GPT_MODEL} procederà con un input testuale vuoto. "
                    "Il risultato potrebbe non essere significativo a meno che il prompt o le istruzioni aggiuntive non gestiscano questo caso."
                )

            main_prompt_instruction = load_main_prompt() # load_main_prompt has st.stop() on failure
            esempi_texts = load_esempi() # Loads examples, with warnings if not found

            with st.spinner(f"Attendere prego: Analisi con {GPT_MODEL} in corso..."):
                esempi_block = "<esempi>\n"
                if esempi_texts:
                    for esempio_text in esempi_texts:
                        # Ensure example text is properly formatted if it contains special XML characters,
                        # but for simple text, direct inclusion is usually fine.
                        esempi_block += f"        <esempio>\n            {esempio_text}\n        </esempio>\n"
                else:
                    esempi_block += "        \n"
                esempi_block += "</esempi>"

                additional_instructions_block = ""
                if user_additional_instructions.strip():
                    additional_instructions_block = (
                        f"\n<additional_instructions>\n{user_additional_instructions.strip()}\n</additional_instructions>\n"
                    )

                # Prepare transcription for GPT input (replace newlines with spaces, as in original logic)
                trascrizione_for_gpt_input = trascrizione_raw.replace("\n", " ").strip() if trascrizione_raw else ""

                full_prompt_for_gpt = f"""{main_prompt_instruction}

{esempi_block}
{additional_instructions_block}
<input>
    {trascrizione_for_gpt_input}
</input>
"""
                if st.checkbox(f"Mostra prompt completo inviato a {GPT_MODEL} (per debug)", False, key="show_prompt_checkbox"):
                    st.text_area(f"Prompt per {GPT_MODEL}:", full_prompt_for_gpt, height=300, key="full_prompt_display")

                gpt_response_obj = gpt_request(GPT_MODEL, full_prompt_for_gpt, 0.8)

                if gpt_response_obj and gpt_response_obj.choices:
                    risultato_gpt = gpt_response_obj.choices[0].message.content.strip()
                    st.subheader(f"📰 Risultato dall'Analisi {GPT_MODEL}:")
                    st.markdown(risultato_gpt) # Display rendered markdown

                    with st.expander("Copia il risultato dell'analisi (testo grezzo)", expanded=False):
                        st.code(risultato_gpt, language='text')
                        st.caption("Testo grezzo dell'analisi. Usa l'icona di copia (solitamente in alto a destra) per copiare.")
                else:
                    st.error(f"Non è stato possibile ottenere una risposta valida da {GPT_MODEL} o la risposta era vuota.")
else:
    st.info("Carica un file audio o video per iniziare.")

st.markdown("---")
st.markdown(
    f"App creata con Streamlit e OpenAI. Modelli usati: **{TRANSCRIPTION_MODEL}** per la trascrizione, "
    f"**{GPT_MODEL}** per l'analisi. Legge il prompt da `prompt.txt` e gli esempi (opzionali) "
    f"dalla directory `esempi/`."
)