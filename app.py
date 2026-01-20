import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- 1. CONFIGURACIÓN VISUAL (ESTILO ORIGINAL) ---
st.set_page_config(page_title="Entrenador Legal TITÁN v6.4", page_icon="⚖️", layout="wide")
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

# --- 2. CEREBRO LÓGICO (TITÁN CASUÍSTICO v6.4) ---
class LegalEngineTITAN:
    def __init__(self):
        self.chunks = []           
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
        else:
            start_idx = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)):
                self.mastery_tracker[start_idx + i] = 0 
        return len(new_chunks)

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
        score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        perc = int((score / (total * 3)) * 100) if total > 0 else 0
        return min(perc, 100), len(self.failed_indices), total

    def get_calibration_prompt(self):
        if not self.feedback_history: return "Modo: Estándar."
        counts = Counter(self.feedback_history)
        instructions = ["🛡️ ANTI-RECORTE: Prohibido 'Solo A' si la norma dice 'A y B'."]
        if counts['desconectado'] > 0: instructions.append("🔗 ANTI-SPOILER: No regales el dato clave.")
        if counts['sesgo_longitud'] > 0: instructions.append("🛑 FORMATO: Opciones de igual longitud visual.")
        if counts['respuesta_obvia'] > 0: instructions.append("💀 DIFICULTAD: Trampas de pertinencia agresivas.")
        if counts['pregunta_facil'] > 0: instructions.append("⚠️ DETALLE: Respuesta basada en datos minúsculos.")
        if counts['alucinacion'] > 0: 
            self.current_temperature = 0.0
            instructions.append("⛔ FUENTE CERRADA: Usa solo el texto provisto.")
        return "\n".join(instructions)

    def generate_case(self):
        if not self.chunks: return {"error": "Carga una norma primero."}
        
        if self.simulacro_mode:
            idx = random.choice(range(len(self.chunks)))
        else:
            if self.failed_indices and random.random() < 0.6:
                idx = random.choice(list(self.failed_indices))
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        chunk = self.chunks[idx]
        
        prompt = f"""
        ACTÚA COMO UN EXPERTO DISEÑADOR CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO: {self.entity if self.entity else 'GENERAL'}.
        TEXTO: "{chunk[:6000]}"
        
        REGLA DE ORO (LA TRAMPA DE PERTINENCIA):
        Opciones incorrectas deben ser leyes reales pero inaplicables AQUÍ.
        
        {self.get_calibration_prompt()}
        
        FORMATO JSON:
        {{
            "narrativa_caso": "Historia detallada...",
            "preguntas": [
                {{
                    "enunciado": "Pregunta...",
                    "opciones": {{ "A": "...", "B": "...", "C": "..." }},
                    "respuesta": "B",
                    "explicacion": "..."
                }}
            ]
        }}
        """
        try:
            res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "temperature": self.current_temperature})
            return json.loads(res.text)
        except:
            return {"error": "Error de comunicación con la IA. Reintenta."}

# --- 3. INICIALIZACIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- 4. INTERFAZ ---
with st.sidebar:
    st.title("⚙️ Panel de Control")
    key = st.text_input("Ingresa tu API Key:", type="password")
    if key and not engine.model:
        ok, msg = engine.configure_api(key)
        if ok: st.success("Conectado")
    
    st.divider()
    engine.level = st.selectbox("Nivel:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    engine.entity = st.text_input("Entidad:", placeholder="Ej: Fiscalía...")
    txt_input = st.text_area("Texto de la Norma:", height=200)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚠️ INICIAR NUEVO", type="primary"):
            if engine.process_law(txt_input):
                st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
    with col2:
        if st.button("➕ AGREGAR"):
            if engine.process_law(txt_input, True): st.success("Agregado.")
            
    if st.button("🗑️ Borrar Calibración"):
        engine.feedback_history = []
        st.toast("Memoria limpia.")

    if engine.chunks:
        save_data = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "log": engine.mistakes_log, "feed": engine.feedback_history, "entity": engine.entity, "level": engine.level})
        st.download_button("Descargar JSON", save_data, "progreso.json")

# --- 5. ÁREA DE JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"<div style='background:#eee; padding:10px; border-radius:8px;'><b>DOMINIO: {perc}%</b> | <b>BLOQUES: {total}</b> | <b style='color:red'>REPASOS: {fails}</b></div>", unsafe_allow_html=True)
    st.progress(float(perc) / 100.0)

    if not st.session_state.get('current_data'):
        with st.spinner("⚖️ Diseñando caso situacional..."):
            data = engine.generate_case()
            if "error" in data:
                st.error(data['error'])
                if st.button("Reintentar"): st.rerun()
                st.stop()
            st.session_state.current_data = data; st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()

    data = st.session_state.current_data
    q_idx = st.session_state.q_idx
    questions = data.get('preguntas', [])

    st.markdown(f"<div class='narrative-box'><h4>📜 Caso: {engine.entity if engine.entity else 'General'}</h4>{data.get('narrativa_caso','')}</div>", unsafe_allow_html=True)

    if q_idx < len(questions):
        q = questions[q_idx]
        st.markdown(f"### 🔹 Pregunta {q_idx + 1}")
        with st.form(key=f"q_form_{q_idx}"):
            sel = st.radio(q['enunciado'], [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
            if st.form_submit_button("✅ Validar Respuesta") and sel:
                if sel[0] == q['respuesta']:
                    st.success("✅ ¡CORRECTO!")
                    if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                else:
                    st.error(f"❌ Era la {q['respuesta']}."); engine.failed_indices.add(engine.current_chunk_idx)
                st.info(f"💡 {q['explicacion']}"); st.session_state.answered = True

        if st.session_state.answered:
            if st.button("⏭️ Siguiente") if q_idx < len(questions)-1 else st.button("🔄 FINALIZAR"):
                if q_idx < len(questions)-1: st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
                else:
                    engine.mastery_tracker[engine.current_chunk_idx] += 1
                    st.session_state.current_data = None; st.rerun()

elif st.session_state.page == 'setup':
    st.title("🏛️ Entrenador Legal TITÁN v6.4")