import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- SOPORTE DEEP LEARNING ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Entrenador Legal TITÁN v7.1", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em;}
    .narrative-box {
        background-color: #f0f4f8; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 8px solid #1e3a8a; 
        margin-bottom: 25px;
        font-family: 'serif';
        font-size: 1.1em;
        line-height: 1.6;
    }
    .status-card {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #d1d5db;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_dl_model():
    if DL_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

nn_model = load_dl_model()

# --- MOTOR LÓGICO ---
class LegalEngineTITAN:
    def __init__(self):
        self.chunks = []
        self.chunk_embeddings = None
        self.mastery_tracker = {}
        self.failed_indices = set()
        self.feedback_history = []
        self.current_data = None
        self.current_chunk_idx = -1
        self.entity = ""
        self.level = "Profesional"
        self.model = None
        self.last_failed_embedding = None

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            return True
        except: return False

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = [text[i:i+5500] for i in range(0, len(text), 5500)]
        
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
        else:
            start = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)): self.mastery_tracker[start+i] = 0
            
        if DL_AVAILABLE and nn_model:
            with st.spinner("🧠 Generando Mapa Neuronal de la Norma..."):
                self.chunk_embeddings = nn_model.encode(self.chunks)
        return len(new_chunks)

    def generate_case(self):
        # Selección inteligente por Deep Learning
        idx = -1
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker[i] < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]

        if idx == -1:
            pending = [i for i, v in self.mastery_tracker.items() if v < 3]
            idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))

        self.current_chunk_idx = idx
        chunk = self.chunks[idx]
        
        # Construcción del Prompt con todas las reglas acumuladas
        calibracion = "\n".join(self.feedback_history)
        
        prompt = f"""
        ACTÚA COMO UN EXPERTO JURISTA PARA EL CARGO: {self.level.upper()}.
        ENTIDAD DEL CASO: {self.entity if self.entity else 'Estado Colombiano'}.
        
        TEXTO BASE: "{chunk}"
        
        INSTRUCCIONES:
        1. Crea un caso situacional donde las opciones incorrectas sean leyes reales pero INAPLICABLES a este hecho específico (Trampa de Pertinencia).
        2. NO menciones el dato clave en el enunciado (Anti-Spoiler).
        3. No recortes la norma en la respuesta correcta (Integridad).
        4. NIVEL {self.level}: Requiere análisis de alta complejidad, no definiciones simples.
        
        {calibracion}
        
        Responde estrictamente en JSON:
        {{
            "narrativa": "Caso detallado...",
            "preguntas": [
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}}
            ]
        }}
        """
        
        try:
            res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(res.text)
        except: return None

# --- APP ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
engine = st.session_state.engine

with st.sidebar:
    st.title("⚖️ TITÁN v7.1")
    st.info("🧠 Deep Learning " + ("Activo" if DL_AVAILABLE else "No detectado"))
    
    key = st.text_input("Gemini API Key:", type="password")
    if key: engine.configure_api(key)
    
    st.divider()
    engine.level = st.selectbox("Nivel del Cargo:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    engine.entity = st.text_input("Entidad (Escenario):", placeholder="Ej: Fiscalía General")
    
    norma = st.text_area("Cargar Texto de la Norma:", height=200)
    if st.button("🚀 Cargar e Iniciar"):
        if engine.process_law(norma):
            st.session_state.data = None
            st.rerun()

# --- ÁREA DE ESTUDIO ---
if engine.chunks:
    if 'data' not in st.session_state or st.session_state.data is None:
        with st.spinner("🧠 La IA está analizando la ley y diseñando el caso..."):
            st.session_state.data = engine.generate_case()
            st.session_state.q_idx = 0
            st.session_state.answered = False

    data = st.session_state.data
    if data:
        # Progreso
        done = sum([min(v, 3) for v in engine.mastery_tracker.values()])
        total = len(engine.chunks) * 3
        st.progress(done/total)
        st.write(f"📊 Dominio: {int((done/total)*100)}% | Nivel: {engine.level}")

        st.markdown(f"<div class='narrative-box'><b>CONTEXTO INSTITUCIONAL: {engine.entity if engine.entity else 'GENERAL'}</b><br>{data['narrativa']}</div>", unsafe_allow_html=True)
        
        q = data['preguntas'][st.session_state.q_idx]
        st.subheader(f"Pregunta {st.session_state.q_idx + 1}")
        st.write(q['enunciado'])
        
        with st.form(key=f"q_{st.session_state.q_idx}"):
            ans = st.radio("Seleccione la opción correcta:", [f"{k}) {v}" for k, v in q['opciones'].items()], index=None)
            submit = st.form_submit_button("Validar Respuesta")
            
            if submit and ans:
                letter = ans[0]
                if letter == q['respuesta']:
                    st.success("✅ ¡CORRECTO!")
                    engine.last_failed_embedding = None
                else:
                    st.error(f"❌ INCORRECTO. La respuesta era la {q['respuesta']}")
                    if DL_AVAILABLE and engine.chunk_embeddings is not None:
                        engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                
                st.info(f"💡 EXPLICACIÓN: {q['explicacion']}")
                st.session_state.answered = True

        if st.session_state.answered:
            if st.button("⏭️ Siguiente"):
                if st.session_state.q_idx < len(data['preguntas']) - 1:
                    st.session_state.q_idx += 1
                    st.session_state.answered = False
                    st.rerun()
                else:
                    engine.mastery_tracker[engine.current_chunk_idx] += 1
                    st.session_state.data = None
                    st.rerun()
                    
        with st.expander("🛠️ Calibrar dificultad"):
            if st.button("⚠️ Más difícil (Trampas de pertinencia)"):
                engine.feedback_history.append("Incrementa la sutileza de los distractores.")
                st.toast("Ajuste guardado para el próximo caso.")
else:
    st.title("Bienvenido a TITÁN v7.1")
    st.write("Carga una norma en el panel izquierdo para comenzar tu entrenamiento neuronal.")