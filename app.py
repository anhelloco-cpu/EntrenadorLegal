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

# --- 1. CONFIGURACIÓN VISUAL (ESTILO ORIGINAL) ---
st.set_page_config(page_title="Entrenador Legal TITÁN v7.1", page_icon="🧠", layout="wide")
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
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

nn_model = load_dl_model()

# --- 2. CEREBRO LÓGICO (ESTRUCTURA ORIGINAL + DEEP LEARNING) ---
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
                with st.spinner("🧠 Generando Mapa Neuronal..."):
                    self.chunk_embeddings = nn_model.encode(self.chunks)
        else:
            start_idx = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)):
                self.mastery_tracker[start_idx + i] = 0 
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
        if counts['desconectado'] > 0: instructions.append("🔗 ANTI-SPOILER: No regales el dato clave.")
        if counts['sesgo_longitud'] > 0: instructions.append("🛑 FORMATO VISUAL: Opciones del mismo largo.")
        if counts['respuesta_obvia'] > 0: instructions.append("💀 DIFICULTAD: Trampas de pertinencia agresivas.")
        if counts['pregunta_facil'] > 0: instructions.append("⚠️ DETALLE: Respuesta basada en datos minúsculos.")
        if counts['repetitivo'] > 0: self.current_temperature = 0.7; instructions.append("🔄 VARIEDAD: Cambia nombres y cargos.")
        if counts['alucinacion'] > 0: self.current_temperature = 0.0; instructions.append("⛔ FUENTE CERRADA: Usa solo el texto.")
        return "\n".join(instructions)

    def generate_case(self):
        if not self.chunks: return {"error": "⚠️ Carga una norma primero."}
        
        idx = -1
        # --- LÓGICA NEURONAL ---
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]

        if idx == -1:
            if self.simulacro_mode: idx = random.choice(range(len(self.chunks)))
            elif self.failed_indices: idx = random.choice(list(self.failed_indices)) if random.random() < 0.6 else random.choice(range(len(self.chunks)))
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        text_chunk = self.chunks[idx]
        
        # --- DEFINICIÓN DE NIVEL ---
        l_msg = ""
        if self.level == "Asistencial": l_msg = "BÁSICO: Pasos y definiciones."
        elif self.level == "Técnico": l_msg = "TÉCNICO: Procesos."
        elif self.level == "Profesional": l_msg = "PROFESIONAL: Análisis y Criterio complejo."
        elif self.level == "Asesor": l_msg = "ASESOR: Estrategia y Riesgo."

        # --- ANCLAJE DE ENTIDAD ---
        contexto_entidad = f"LA HISTORIA DEBE OCURRIR OBLIGATORIAMENTE EN: {self.entity.upper()}." if self.entity else "Escenario institucional genérico."

        prompt = f"""
        ACTÚA COMO UN EXPERTO JURISTA CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO: {contexto_entidad}
        TEXTO NORMATIVO: "{text_chunk[:6000]}"
        
        REGLAS:
        1. TRAMPA DE PERTINENCIA: Opciones incorrectas son leyes reales pero inaplicables AQUÍ.
        2. ANTI-SPOILER: No reveles el dato clave en la pregunta.
        3. INTEGRIDAD: Respuesta correcta completa, no resumida.
        4. NIVEL {self.level}: {l_msg}
        
        AJUSTES: {self.get_calibration_prompt()}
        
        FORMATO JSON:
        {{
            "narrativa_caso": "Caso detallado en {self.entity if self.entity else 'la entidad'}...",
            "preguntas": [
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}},
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}},
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}},
                {{"enunciado": "...", "opciones": {{"A": "...", "B": "...", "C": "..."}}, "respuesta": "A", "explicacion": "..."}}
            ]
        }}
        """
        
        try:
            res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(res.text)
        except: return {"error": "Fallo en IA"}

# --- 3. INICIALIZACIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- 4. INTERFAZ (RESTURADA) ---
with st.sidebar:
    st.title("⚙️ Panel TITÁN v7.1")
    if DL_AVAILABLE: st.success("🧠 Deep Learning: ON")
    
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
                st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
    with col2:
        if st.button("➕ AGREGAR"):
            if engine.process_law(txt_input, True): st.success("Agregado.")

    st.divider()
    if st.button("🗑️ Borrar Calibración"):
        engine.feedback_history = []
        st.toast("Memoria limpia.")

    if engine.chunks:
        save_data = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "log": engine.mistakes_log, "feed": engine.feedback_history, "entity": engine.entity})
        st.download_button("Descargar Progreso", save_data, "progreso.json")

# --- 5. JUEGO (RESTAURADO) ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"<div style='background:#eee; padding:10px; border-radius:8px;'><b>DOMINIO: {perc}%</b> | <b>BLOQUES: {total}</b> | <b>REPASOS: {fails}</b></div>", unsafe_allow_html=True)
    st.progress(perc/100)

    if not st.session_state.current_data:
        with st.spinner("🧠 Neurona analizando debilidades y diseñando caso..."):
            data = engine.generate_case()
            if "error" in data: st.error("Fallo IA"); st.stop()
            st.session_state.current_data = data; st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()

    data = st.session_state.current_data
    st.markdown(f"<div class='narrative-box'><h4>📜 Caso: {engine.entity if engine.entity else 'General'}</h4>{data['narrativa_caso']}</div>", unsafe_allow_html=True)

    q = data['preguntas'][st.session_state.q_idx]
    st.subheader(f"Pregunta {st.session_state.q_idx + 1}")
    st.write(q['enunciado'])
    
    with st.form(key=f"q_{st.session_state.q_idx}"):
        sel = st.radio("Opciones:", [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
        if st.form_submit_button("Validar Respuesta") and sel:
            if sel[0] == q['respuesta']:
                st.success("✅ ¡CORRECTO!")
                if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                engine.last_failed_embedding = None
            else:
                st.error(f"❌ INCORRECTO. Era {q['respuesta']}")
                engine.failed_indices.add(engine.current_chunk_idx)
                if DL_AVAILABLE and engine.chunk_embeddings is not None:
                    engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
            st.info(f"💡 {q['explicacion']}"); st.session_state.answered = True

    if st.session_state.answered:
        col_nav, col_rep = st.columns(2)
        with col_nav:
            if st.button("⏭️ Siguiente"):
                if st.session_state.q_idx < 3: st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
                else: engine.mastery_tracker[engine.current_chunk_idx] += 1; st.session_state.current_data = None; st.rerun()
        with col_rep:
            with st.expander("📢 Reportar Fallo"):
                report = st.selectbox("Error:", ["Spoiler", "Obvio", "Incompleto"])
                if st.button("Calibrar"):
                    engine.feedback_history.append(report); st.toast("Ajustado.")

elif st.session_state.page == 'setup':
    st.title("🧠 Entrenador TITÁN v7.1")