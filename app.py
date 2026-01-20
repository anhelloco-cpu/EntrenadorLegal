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

# --- 1. CONFIGURACIÓN VISUAL RESTAURADA ---
st.set_page_config(page_title="Entrenador Legal TITÁN v7.4", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s;}
    .narrative-box {
        background-color: #e8f5e9; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 6px solid #2e7d32; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        font-family: 'Georgia', serif;
        font-size: 1.15em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_dl_model():
    if DL_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

nn_model = load_dl_model()

class LegalEngineTITAN:
    def __init__(self):
        self.chunks = []           
        self.chunk_embeddings = None
        self.mastery_tracker = {}  
        self.failed_indices = set()
        self.mistakes_log = []
        self.feedback_history = []
        self.current_data = None
        self.current_chunk_idx = -1
        self.entity = ""
        self.level = "Profesional" 
        self.model = None
        self.current_temperature = 0.1 # Temperatura baja para mayor precisión JSON
        self.last_failed_embedding = None 

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            return True, "Conectado"
        except Exception as e:
            return False, str(e)

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = [text[i:i+5500] for i in range(0, len(text), 5500)]
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            if DL_AVAILABLE and nn_model:
                with st.spinner("🧠 Generando Mapa Neuronal..."):
                    self.chunk_embeddings = nn_model.encode(self.chunks)
        else:
            start_idx = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)): self.mastery_tracker[start_idx + i] = 0 
            if DL_AVAILABLE and nn_model:
                self.chunk_embeddings = nn_model.encode(self.chunks)
        return len(new_chunks)

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
        score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        perc = int((score / (total * 3)) * 100) if total > 0 else 0
        return min(perc, 100), len(self.failed_indices), total

    def generate_case(self):
        if not self.chunks: return {"error": "Carga una norma."}
        
        # Selección Neuronal
        idx = -1
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            cands = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            cands.sort(key=lambda x: x[1], reverse=True)
            if cands: idx = cands[0][0]

        if idx == -1:
            pending = [k for k,v in self.mastery_tracker.items() if v < 3]
            idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        chunk = self.chunks[idx]
        
        prompt = f"""
        Actúa como experto jurista. Nivel: {self.level.upper()}. Escenario: {self.entity if self.entity else 'General'}.
        Usa este texto: "{chunk[:5000]}"
        REGLAS: 
        1. Opciones incorrectas: Leyes reales pero inaplicables.
        2. No reveles el dato clave en el enunciado.
        3. Respuesta correcta completa.
        
        RESPONDE EXCLUSIVAMENTE EN FORMATO JSON:
        {{
            "narrativa_caso": "Historia detallada...",
            "preguntas": [
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}}
            ]
        }}
        """
        
        for intento in range(3): # Aumentamos a 3 intentos
            try:
                res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                return json.loads(res.text)
            except Exception:
                if intento == 2: return {"error": "La IA está saturada. Intenta con un fragmento de norma más corto."}
                time.sleep(1.5)
        return {"error": "Fallo crítico."}

# --- SESIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ TITÁN v7.4")
    key = st.text_input("Gemini API Key:", type="password")
    if key and not engine.model: engine.configure_api(key)
    st.divider()
    engine.level = st.selectbox("Nivel:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    engine.entity = st.text_input("Entidad:", placeholder="Ej: Fiscalía")
    txt = st.text_area("Cargar Norma:", height=200)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 INICIAR", type="primary"):
            if engine.process_law(txt): st.session_state.page = 'game'; st.session_state.data = None; st.rerun()
    with col2:
        if st.button("➕ SUMAR"):
            if engine.process_law(txt, True): st.success("Sumado")
    
    st.divider()
    if engine.chunks:
        data_save = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "entity": engine.entity, "level": engine.level})
        st.download_button("📥 Descargar Progreso", data_save, "progreso.json")

# --- ÁREA DE JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"**DOMINIO: {perc}%** | **BLOQUES: {total}** | **REPASOS: {fails}**")
    st.progress(perc/100)

    if not st.session_state.get('data'):
        with st.spinner("🧠 Diseñando caso neuronal..."):
            st.session_state.data = engine.generate_case()
            st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()

    data = st.session_state.data
    if "error" in data:
        st.error(data["error"])
        if st.button("🔄 Reintentar"): st.session_state.data = None; st.rerun()
    else:
        st.markdown(f"<div class='narrative-box'><h4>📜 Caso: {engine.entity if engine.entity else 'General'}</h4>{data['narrativa_caso']}</div>", unsafe_allow_html=True)
        q = data['preguntas'][st.session_state.q_idx]
        st.subheader(f"Pregunta {st.session_state.q_idx + 1}")
        st.write(q['enunciado'])
        
        with st.form(key=f"q_{st.session_state.q_idx}"):
            sel = st.radio("Opciones:", [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
            if st.form_submit_button("Validar"):
                if sel and sel[0] == q['respuesta']:
                    st.success("✅ ¡CORRECTO!")
                    if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                    engine.last_failed_embedding = None
                else:
                    st.error(f"❌ Era la {q['respuesta']}")
                    engine.failed_indices.add(engine.current_chunk_idx)
                    if DL_AVAILABLE and engine.chunk_embeddings is not None:
                        engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                st.info(q['explicacion']); st.session_state.answered = True

        if st.session_state.answered:
            if st.button("⏭️ Siguiente"):
                if st.session_state.q_idx < len(data['preguntas']) - 1:
                    st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
                else:
                    engine.mastery_tracker[engine.current_chunk_idx] += 1
                    st.session_state.data = None; st.rerun()

elif st.session_state.page == 'setup':
    st.title("🧠 Entrenador TITÁN v7.4")