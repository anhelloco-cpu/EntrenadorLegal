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

# --- 1. CONFIGURACIÓN VISUAL (RESTAURADA) ---
st.set_page_config(page_title="Entrenador Legal TITÁN v7.3", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.02);}
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
    .status-bar {font-weight: bold; color: #2e86c1;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_dl_model():
    if DL_AVAILABLE:
        # Cargamos el modelo neuronal de 384 dimensiones
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

nn_model = load_dl_model()

# --- 2. MOTOR LÓGICO TITÁN ---
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
            return False, str(e)

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = [text[i:i+5500] for i in range(0, len(text), 5500)]
        
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            self.mistakes_log = []
            if DL_AVAILABLE and nn_model:
                with st.spinner("🧠 Mapeando semánticamente la norma con Deep Learning..."):
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
        total_chunks = len(self.chunks)
        goal_score = total_chunks * 3 
        current_score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        percentage = int((current_score / goal_score) * 100) if goal_score > 0 else 0
        return min(percentage, 100), len(self.failed_indices), total_chunks

    def get_calibration_prompt(self):
        if not self.feedback_history: return "Modo: Estándar."
        counts = Counter(self.feedback_history)
        instructions = []
        instructions.append("🛡️ ANTI-RECORTE: Prohibido 'Solo A' si la norma dice 'A y B'.")
        if "Spoiler" in counts: instructions.append("🔗 ANTI-SPOILER: No reveles el dato clave en la pregunta.")
        if "Obvio" in counts: instructions.append("💀 DIFICULTAD: Trampas de pertinencia agresivas.")
        if "Incompleto" in counts: instructions.append("⚠️ DETALLE: La respuesta debe ser exhaustiva.")
        return "\n".join(instructions)

    def generate_case(self):
        if not self.chunks: return {"error": "Carga una norma."}
        
        idx = -1
        # --- LÓGICA NEURONAL DE SELECCIÓN ---
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: 
                idx = candidatos[0][0]
                st.toast("🧠 Radar Neuronal detectó una debilidad conceptual.", icon="🕵️")

        if idx == -1:
            if self.failed_indices:
                idx = random.choice(list(self.failed_indices)) if random.random() < 0.6 else random.choice(range(len(self.chunks)))
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        chunk = self.chunks[idx]
        
        prompt = f"""
        ACTÚA COMO UN EXPERTO JURISTA CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO: {self.entity.upper() if self.entity else 'GENERAL'}.
        TEXTO: "{chunk[:6000]}"
        
        REGLAS:
        1. TRAMPA DE PERTINENCIA: Opciones incorrectas son leyes reales pero inaplicables AQUÍ.
        2. ANTI-SPOILER: No reveles el dato clave en la pregunta.
        3. INTEGRIDAD: Respuesta correcta completa, no resumida.
        
        AJUSTES: {self.get_calibration_prompt()}
        
        FORMATO JSON (OBLIGATORIO):
        {{
            "narrativa_caso": "Historia detallada...",
            "preguntas": [
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}},
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}},
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}},
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}}
            ]
        }}
        """
        
        for intento in range(2):
            try:
                res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                return json.loads(res.text)
            except:
                if intento == 1: return {"error": "Fallo IA crítico tras reintento."}
                time.sleep(2)
        return {"error": "Fallo IA."}

# --- 3. INICIALIZACIÓN DE SESIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- 4. PANEL LATERAL (ESTILO v6.4 RESTAURADO) ---
with st.sidebar:
    st.title("⚙️ TITÁN v7.3 (Neuronal)")
    if DL_AVAILABLE: st.success("🧠 Deep Learning ON")
    
    key = st.text_input("Gemini API Key:", type="password")
    if key and not engine.model: engine.configure_api(key)
    
    st.divider()
    engine.level = st.selectbox("Nivel del Cargo:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    engine.entity = st.text_input("Entidad (Escenario):", placeholder="Ej: Fiscalía General")
    
    txt_input = st.text_area("Cargar Norma:", height=200)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 INICIAR NUEVO", type="primary"):
            if engine.process_law(txt_input):
                st.session_state.page = 'game'; st.session_state.data = None; st.rerun()
    with col2:
        if st.button("➕ AGREGAR"):
            if engine.process_law(txt_input, True): st.success("Norma sumada.")

    st.divider()
    if st.button("🗑️ Borrar Calibración"):
        engine.feedback_history = []
        st.toast("Memoria limpia.")

    if engine.chunks:
        save_data = json.dumps({
            "chunks": engine.chunks, 
            "mastery": engine.mastery_tracker, 
            "failed": list(engine.failed_indices), 
            "log": engine.mistakes_log, 
            "feed": engine.feedback_history, 
            "entity": engine.entity,
            "level": engine.level
        })
        st.download_button("📥 Descargar JSON", save_data, "progreso_titan.json")

    upl = st.file_uploader("📤 Cargar JSON", type=['json'])
    if upl:
        try:
            d = json.load(upl)
            engine.chunks = d['chunks']
            engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
            engine.failed_indices = set(d['failed'])
            engine.mistakes_log = d['log']
            engine.feedback_history = d['feed']
            engine.entity = d['entity']
            engine.level = d['level']
            st.success("¡Progreso recuperado!")
        except: st.error("Archivo inválido.")

# --- 5. ÁREA DE JUEGO (ESTILO v6.4 RESTAURADO) ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"""
    <div style='background:#eee; padding:15px; border-radius:10px; display:flex; justify-content:space-between; align-items:center;'>
        <span class='status-bar'>DOMINIO: {perc}%</span>
        <span class='status-bar'>REPASOS PENDIENTES: {fails}</span>
        <span class='status-bar'>BLOQUES: {total}</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(perc/100)

    if not st.session_state.get('data'):
        with st.spinner("🧠 El radar neuronal está buscando debilidades semánticas y diseñando el caso..."):
            st.session_state.data = engine.generate_case()
            st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()

    data = st.session_state.data
    if "error" in data:
        st.error(data["error"])
        if st.button("🔄 Reintentar"): st.session_state.data = None; st.rerun()
    else:
        st.markdown(f"<div class='narrative-box'><h4>📜 Caso Situacional: {engine.entity if engine.entity else 'General'}</h4>{data['narrativa_caso']}</div>", unsafe_allow_html=True)
        
        q = data['preguntas'][st.session_state.q_idx]
        st.subheader(f"Pregunta {st.session_state.q_idx + 1} de 4")
        st.markdown(f"**{q['enunciado']}**")
        
        with st.form(key=f"q_form_{st.session_state.q_idx}"):
            sel = st.radio("Seleccione su respuesta:", [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
            if st.form_submit_button("✅ VALIDAR RESPUESTA") and sel:
                letter = sel[0]
                if letter == q['respuesta']:
                    st.success("✨ ¡CORRECTO!")
                    if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                    engine.last_failed_embedding = None
                else:
                    st.error(f"❌ INCORRECTO. La respuesta era la {q['respuesta']}")
                    engine.failed_indices.add(engine.current_chunk_idx)
                    engine.mistakes_log.append({"pregunta": q['enunciado'], "error": letter, "correcta": q['respuesta']})
                    # --- DEEP LEARNING: GUARDAMOS EL VECTOR DEL ERROR ---
                    if DL_AVAILABLE and engine.chunk_embeddings is not None:
                        engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                
                st.info(f"💡 EXPLICACIÓN: {q['explicacion']}")
                st.session_state.answered = True

        if st.session_state.answered:
            col_nav, col_rep = st.columns(2)
            with col_nav:
                if st.session_state.q_idx < 3:
                    if st.button("⏭️ Siguiente Pregunta"):
                        st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
                else:
                    if st.button("🔄 FINALIZAR CASO"):
                        engine.mastery_tracker[engine.current_chunk_idx] += 1
                        st.session_state.data = None; st.rerun()
            with col_rep:
                with st.expander("📢 Calibrar dificultad"):
                    report = st.selectbox("Reportar:", ["Spoiler", "Obvio", "Incompleto"])
                    if st.button("Guardar Ajuste"):
                        engine.feedback_history.append(report); st.toast("Ajuste recibido.")

elif st.session_state.page == 'setup':
    st.title("🧠 Bienvenido a TITÁN v7.3")
    st.write("Configura tu sesión en el panel lateral para comenzar el entrenamiento situacional.")