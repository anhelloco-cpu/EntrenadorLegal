import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- NUEVAS LIBRERÍAS DE DEEP LEARNING ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# --- 1. CONFIGURACIÓN VISUAL ROBUSTA ---
st.set_page_config(page_title="Entrenador Legal TITÁN v7.1 (Neuronal)", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.02);}
    .narrative-box {
        background-color: #e0f7fa; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 6px solid #006064; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        font-family: 'Georgia', serif;
        font-size: 1.15em;
    }
    .question-card {
        background-color: #ffffff;
        padding: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .status-bar {font-weight: bold; color: #2e86c1;}
</style>
""", unsafe_allow_html=True)

# --- CACHÉ DEL MODELO NEURONAL ---
@st.cache_resource
def load_embedding_model():
    if DL_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

dl_model = load_embedding_model()

# --- LISTA DE ENTIDADES PÚBLICAS COLOMBIA ---
ENTIDADES_CO = [
    "Fiscalía General de la Nación", "Procuraduría General", "DIAN", 
    "Contraloría General", "Policía Nacional", "Ejército Nacional",
    "Ministerio de Educación", "Ministerio de Salud", "ICBF", 
    "SENA", "Rama Judicial", "Registraduría Nacional", "DANE", "Otra (Manual)"
]

# --- 2. CEREBRO LÓGICO ---
class LegalEngineTITAN:
    def __init__(self):
        self.chunks = []           
        self.chunk_embeddings = None 
        self.chunk_origins = {}    
        self.mastery_tracker = {}  
        self.failed_indices = set()
        self.mistakes_log = []
        self.feedback_history = []
        self.current_data = None
        self.current_chunk_idx = -1
        self.entity = ""
        self.level = "Profesional" 
        self.simulacro_mode = False
        self.model = None
        self.current_temperature = 0.2 
        self.last_failed_embedding = None 

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            return True, "Conectado"
        except Exception as e:
            return False, f"Error Crítico API: {str(e)}"

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = []
        step = 5500 
        for i in range(0, len(text), step):
            chunk = text[i:i+step]
            new_chunks.append(chunk)
            
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            self.mistakes_log = []
            if dl_model:
                with st.spinner("🧠 Neurona Activada: Comprendiendo norma..."):
                    self.chunk_embeddings = dl_model.encode(self.chunks)
        else:
            start_index = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)):
                self.mastery_tracker[start_index + i] = 0 
            if dl_model:
                with st.spinner("🧠 Neurona Activada: Re-indexando..."):
                    self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(new_chunks)

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
        current_score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        percentage = int((current_score / (total * 3)) * 100) if total > 0 else 0
        return min(percentage, 100), len(self.failed_indices), total

    def get_calibration_prompt(self):
        if not self.feedback_history: return "Modo: Estándar."
        counts = Counter(self.feedback_history)
        instructions = ["🛡️ ANTI-RECORTE: Prohibido 'Solo A' si la norma dice 'A y B'."]
        if counts['desconectado'] > 0: instructions.append("🔗 ANTI-SPOILER: No regales el dato clave.")
        if counts['respuesta_obvia'] > 0: instructions.append("💀 DIFICULTAD: Trampas de pertinencia sutiles.")
        if counts['alucinacion'] > 0: self.current_temperature = 0.0; instructions.append("⛔ FUENTE CERRADA: Usa SOLO el texto.")
        return "\n".join(instructions)

    def generate_case(self):
        if not self.chunks: return {"error": "⚠️ Carga una norma primero."}
        idx = -1
        selection_reason = "Aleatorio"

        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and len(self.failed_indices) == 0:
            similitudes = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(similitudes) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos:
                idx = random.choice(candidatos[:3])[0]
                selection_reason = "Deep Learning: Tema Relacionado a tu último fallo"

        if idx == -1:
            if self.failed_indices:
                idx = random.choice(list(self.failed_indices)) if random.random() < 0.6 else random.choice(range(len(self.chunks)))
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        self.selection_reason = selection_reason
        
        prompt = f"""
        ACTÚA COMO UN EXPERTO JURISTA CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO OBLIGATORIO: {self.entity.upper() if self.entity else 'GENERAL'}.
        TEXTO: "{self.chunks[idx][:6000]}"
        
        INSTRUCCIÓN: Crea un CASO SITUACIONAL donde los hechos ocurran en {self.entity}.
        Las opciones incorrectas deben ser leyes reales pero inaplicables AQUÍ.
        {self.get_calibration_prompt()}
        
        JSON:
        {{
            "narrativa_caso": "Historia detallada en {self.entity}...",
            "preguntas": [
                {{"enunciado": "..", "opciones": {{"A": "..", "B": "..", "C": ".."}}, "respuesta": "B", "explicacion": ".."}}
            ]
        }}
        """
        try:
            res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "temperature": self.current_temperature})
            return json.loads(res.text)
        except: return {"error": "Error de conexión IA."}

# --- 3. INICIALIZACIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- 4. INTERFAZ ---
with st.sidebar:
    st.title("⚙️ Panel de Control")
    if DL_AVAILABLE: st.success("🧠 Deep Learning: ACTIVADO")
    
    key = st.text_input("Ingresa tu API Key:", type="password")
    if key and not engine.model:
        ok, msg = engine.configure_api(key)
        if ok: st.success(msg)
    
    st.divider()
    engine.level = st.selectbox("Nivel:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    
    # --- SELECTOR DE ENTIDAD ---
    ent_select = st.selectbox("Entidad Pública Colombia:", ENTIDADES_CO)
    if ent_select == "Otra (Manual)":
        engine.entity = st.text_input("✏️ Escribe el nombre de la Entidad:")
    else:
        engine.entity = ent_select

    txt_input = st.text_area("Texto de la Norma:", height=150)
    
    if st.button("🚀 INICIAR", type="primary"):
        if engine.process_law(txt_input):
            st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
    
    if st.button("➕ SUMAR NORMA"):
        if engine.process_law(txt_input, append=True): st.success("Agregado.")

    if engine.chunks:
        save_data = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "feed": engine.feedback_history, "entity": engine.entity})
        st.download_button("Descargar JSON", save_data, "progreso.json")

# --- 5. JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"**DOMINIO: {perc}%** | **BLOQUES: {total}** | **REPASOS: {fails}**")
    st.progress(perc / 100.0)

    if not st.session_state.get('current_data'):
        with st.spinner("🧠 Radar Neuronal analizando..."):
            st.session_state.current_data = engine.generate_case()
            st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()

    data = st.session_state.current_data
    st.markdown(f"<div class='narrative-box'><h4>📜 Caso: {engine.entity}</h4>{data.get('narrativa_caso','')}</div>", unsafe_allow_html=True)
    
    q = data['preguntas'][st.session_state.q_idx]
    st.write(f"### Pregunta {st.session_state.q_idx + 1}")
    with st.form(key=f"q_{st.session_state.q_idx}"):
        sel = st.radio(q['enunciado'], [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
        if st.form_submit_button("✅ Validar"):
            if sel and sel[0] == q['respuesta']:
                st.success("¡CORRECTO!"); engine.last_failed_embedding = None
                if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
            else:
                st.error(f"Incorrecto. Era la {q['respuesta']}"); engine.failed_indices.add(engine.current_chunk_idx)
                if DL_AVAILABLE and engine.chunk_embeddings is not None:
                    engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
            st.info(q['explicacion']); st.session_state.answered = True

    if st.session_state.answered:
        if st.button("Siguiente ⏭️") if st.session_state.q_idx < 3 else st.button("🔄 FINALIZAR"):
            if st.session_state.q_idx < 3:
                st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
            else:
                engine.mastery_tracker[engine.current_chunk_idx] += 1
                st.session_state.current_data = None; st.rerun()

elif st.session_state.page == 'setup':
    st.title("🧠 Entrenador TITÁN v7.1")