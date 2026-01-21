import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- LIBRERÍAS DE DEEP LEARNING ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="TITÁN v7.1 (Entidades)", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em;}
    .narrative-box {
        background-color: #e0f7fa; padding: 25px; border-radius: 12px; 
        border-left: 6px solid #006064; margin-bottom: 25px;
        font-family: 'Georgia', serif; font-size: 1.15em;
    }
    .status-bar {font-weight: bold; color: #2e86c1;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    if DL_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

dl_model = load_embedding_model()

# --- MOTOR LÓGICO ---
class LegalEngineTITAN:
    def __init__(self):
        self.chunks = []
        self.chunk_embeddings = None
        self.mastery_tracker = {}
        self.failed_indices = set()
        self.feedback_history = []
        self.entity = ""
        self.level = "Profesional"
        self.model = None
        self.last_failed_embedding = None

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            return True, "Conectado"
        except: return False, "Error API"

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = [text[i:i+5500] for i in range(0, len(text), 5500)]
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            if dl_model: self.chunk_embeddings = dl_model.encode(self.chunks)
        else:
            start = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)): self.mastery_tracker[start+i] = 0
            if dl_model: self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(new_chunks)

    def generate_case(self):
        if not self.chunks: return None
        
        # Lógica de Selección Neuronal
        idx = -1
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]
        
        if idx == -1:
            pending = [k for k,v in self.mastery_tracker.items() if v < 3]
            idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        
        # --- PROMPT REFORZADO CON ENTIDAD ---
        prompt = f"""
        ACTÚA COMO UN EXPERTO JURISTA PARA EL NIVEL: {self.level.upper()}.
        ESCENARIO OBLIGATORIO: {self.entity.upper()}.
        NORMA: "{self.chunks[idx][:6000]}"
        
        INSTRUCCIONES CRÍTICAS:
        1. Toda la narrativa debe ocurrir dentro de las funciones y dependencias de la entidad: {self.entity}.
        2. Los distractores (opciones incorrectas) deben ser leyes reales colombianas que NO aplican a este caso específico.
        3. No regales el dato clave.
        
        FORMATO JSON:
        {{
            "narrativa": "Caso situado en {self.entity}...",
            "preguntas": [
                {{"enunciado": "..", "opciones": {{"A": "..", "B": "..", "C": ".."}}, "respuesta": "A", "explicacion": ".."}}
            ]
        }}
        """
        try:
            res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(res.text)
        except: return None

# --- APP ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
engine = st.session_state.engine

# --- ENTIDADES PÚBLICAS COLOMBIA ---
ENTIDADES_CO = [
    "Fiscalía General de la Nación", "Procuraduría General", "DIAN", 
    "Contraloría General", "Policía Nacional", "Ejército Nacional",
    "Ministerio de Educación", "Ministerio de Salud", "ICBF", 
    "SENA", "Rama Judicial", "Registraduría Nacional", "DANE", "Otra (Manual)"
]

with st.sidebar:
    st.title("⚙️ Control TITÁN v7.1")
    key = st.text_input("API Key:", type="password")
    if key: engine.configure_api(key)
    
    st.divider()
    engine.level = st.selectbox("Nivel:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    
    # --- SELECTOR DE ENTIDAD + OPCIÓN MANUAL ---
    ent_select = st.selectbox("Entidad Pública:", ENTIDADES_CO)
    if ent_select == "Otra (Manual)":
        engine.entity = st.text_input("✏️ Escribe el nombre de la Entidad:")
    else:
        engine.entity = ent_select

    txt = st.text_area("Cargar Norma:", height=200)
    if st.button("🚀 INICIAR"):
        if engine.process_law(txt):
            st.session_state.data = None
            st.session_state.page = 'game'; st.rerun()

# --- JUEGO ---
if st.session_state.get('page') == 'game':
    done = sum([min(v, 3) for v in engine.mastery_tracker.values()])
    total = len(engine.chunks) * 3
    st.markdown(f"**ENTIDAD:** {engine.entity.upper()} | **PROGRESO:** {int((done/total)*100)}%")
    st.progress(done/total if total > 0 else 0)

    if not st.session_state.get('data'):
        with st.spinner(f"🧠 Adaptando norma a la realidad de {engine.entity}..."):
            st.session_state.data = engine.generate_case()
            st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()

    data = st.session_state.data
    if data:
        st.markdown(f"<div class='narrative-box'><b>ESCENARIO: {engine.entity}</b><br>{data['narrativa']}</div>", unsafe_allow_html=True)
        q = data['preguntas'][st.session_state.q_idx]
        st.write(f"### Pregunta {st.session_state.q_idx + 1}")
        st.write(q['enunciado'])
        
        with st.form(key=f"q_{st.session_state.q_idx}"):
            sel = st.radio("Opciones:", [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
            if st.form_submit_button("Validar"):
                if sel and sel[0] == q['respuesta']:
                    st.success("✅ ¡CORRECTO!")
                    engine.last_failed_embedding = None
                else:
                    st.error(f"❌ INCORRECTO. Era la {q['respuesta']}")
                    if DL_AVAILABLE and engine.chunk_embeddings is not None:
                        engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                st.info(q['explicacion']); st.session_state.answered = True

        if st.session_state.answered:
            if st.button("Siguiente ⏭️"):
                if st.session_state.q_idx < len(data['preguntas']) - 1:
                    st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
                else:
                    engine.mastery_tracker[engine.current_chunk_idx] += 1
                    st.session_state.data = None; st.rerun()