import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- NUEVAS LIBRERÍAS DE DEEP LEARNING (Opcionales) ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# --- 1. CONFIGURACIÓN VISUAL ROBUSTA ---
st.set_page_config(page_title="TITÁN v7.3 - Full Blindado", page_icon="🇨🇴", layout="wide")
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

# --- LISTA OFICIAL DE ENTIDADES COLOMBIA (CORREGIDA) ---
ENTIDADES_CO = [
    "Contraloría General de la República",
    "Fiscalía General de la Nación",
    "Procuraduría General de la Nación",
    "Defensoría del Pueblo",
    "DIAN (Dirección de Impuestos y Aduanas Nacionales)",
    "Registraduría Nacional del Estado Civil",
    "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia",
    "Consejo de Estado",
    "Corte Constitucional",
    "Policía Nacional de Colombia",
    "Ejército Nacional de Colombia",
    "ICBF (Instituto Colombiano de Bienestar Familiar)",
    "SENA (Servicio Nacional de Aprendizaje)",
    "Ministerio de Educación Nacional",
    "Ministerio de Salud y Protección Social",
    "DANE",
    "Otra (Manual) / Agregar +"
]

# --- 2. CEREBRO LÓGICO COMPLETO ---
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
        new_chunks = [text[i:i+5500] for i in range(0, len(text), 5500)]
        
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

        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and len(self.failed_indices) == 0 and not self.simulacro_mode:
            similitudes = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(similitudes) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos:
                idx = random.choice(candidatos[:3])[0]
                selection_reason = "Deep Learning: Tema Relacionado a tu último fallo"

        if idx == -1:
            if self.simulacro_mode: idx = random.choice(range(len(self.chunks)))
            elif self.failed_indices:
                idx = random.choice(list(self.failed_indices)) if random.random() < 0.6 else random.choice(range(len(self.chunks)))
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        self.selection_reason = selection_reason
        
        prompt = f"""
        ACTÚA COMO UN EXPERTO JURISTA CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO OBLIGATORIO: {self.entity.upper()}.
        TEXTO NORMATIVO: "{self.chunks[idx][:6000]}"
        
        INSTRUCCIÓN: Crea un CASO SITUACIONAL donde los hechos ocurran en {self.entity}.
        Las opciones incorrectas deben ser leyes reales pero inaplicables AQUÍ (Trampa de Pertinencia).
        {self.get_calibration_prompt()}
        
        IMPORTANTE: Responde SOLO con un JSON válido que tenga las claves "narrativa_caso" y "preguntas".
        
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
    st.title("⚙️ TITÁN v7.3")
    if DL_AVAILABLE: st.success("🧠 Deep Learning: ACTIVADO")
    
    key = st.text_input("API Key:", type="password")
    if key and not engine.model:
        ok, msg = engine.configure_api(key)
        if ok: st.success(msg)
    
    st.divider()
    engine.level = st.selectbox("Nivel:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    
    # --- SELECTOR DE ENTIDAD ---
    ent_select = st.selectbox("Entidad Pública:", ENTIDADES_CO)
    if "Otra" in ent_select or "Agregar" in ent_select:
        engine.entity = st.text_input("✏️ Escribe el nombre de la Entidad:")
    else:
        engine.entity = ent_select

    txt_input = st.text_area("Texto de la Norma:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 INICIAR NUEVO", type="primary"):
            if engine.process_law(txt_input):
                st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
    with col2:
        if st.button("➕ AGREGAR"):
            if engine.process_law(txt_input, append=True): st.success("Agregado.")
    
    st.divider()
    if st.button("🔥 MODO SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True; st.session_state.current_data = None; st.session_state.page = 'game'; st.rerun()

    # --- PERSISTENCIA DE DATOS (RECUPERADA) ---
    if engine.chunks:
        save_data = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "feed": engine.feedback_history, "entity": engine.entity})
        st.download_button("Descargar JSON", save_data, "progreso.json")
    
    upl = st.file_uploader("Cargar JSON", type=['json'])
    if upl:
        try:
            d = json.load(upl)
            engine.chunks = d['chunks']
            engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
            engine.failed_indices = set(d['failed'])
            engine.feedback_history = d.get('feed', [])
            engine.entity = d.get('entity', "")
            st.success("¡Progreso cargado!")
        except: st.error("Archivo corrupto.")

# --- 5. JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"**DOMINIO: {perc}%** | **BLOQUES: {total}** | **REPASOS: {fails}**")
    st.progress(perc / 100.0)

    # --- PROTECCIÓN CONTRA KEYERROR (NUEVA) ---
    if not st.session_state.get('current_data'):
        with st.spinner("🧠 Generando caso neuronal..."):
            data = engine.generate_case()
            if data and "preguntas" in data:
                st.session_state.current_data = data
                st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()
            else:
                st.warning("⚠️ La IA tuvo un lapsus. Reintentando...")
                time.sleep(1) # Pequeña pausa
                st.rerun() # Reintento automático

    data = st.session_state.current_data
    st.markdown(f"<div class='narrative-box'><h4>🏛️ Entidad: {engine.entity}</h4>{data.get('narrativa_caso','Error en narrativa')}</div>", unsafe_allow_html=True)
    
    try:
        q = data['preguntas'][st.session_state.q_idx]
        st.write(f"### Pregunta {st.session_state.q_idx + 1}")
        with st.form(key=f"q_{st.session_state.q_idx}"):
            sel = st.radio(q['enunciado'], [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
            if st.form_submit_button("✅ Validar Respuesta"):
                if sel and sel[0] == q['respuesta']:
                    st.success("✅ ¡CORRECTO!"); engine.last_failed_embedding = None
                    if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                else:
                    st.error(f"❌ Era la {q['respuesta']}"); engine.failed_indices.add(engine.current_chunk_idx)
                    if DL_AVAILABLE and engine.chunk_embeddings is not None:
                        engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                st.info(q['explicacion']); st.session_state.answered = True

        if st.session_state.answered:
            if st.button("Siguiente ⏭️") if st.session_state.q_idx < len(data['preguntas'])-1 else st.button("🔄 FINALIZAR CASO"):
                if st.session_state.q_idx < len(data['preguntas'])-1:
                    st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
                else:
                    engine.mastery_tracker[engine.current_chunk_idx] += 1
                    st.session_state.current_data = None; st.rerun()
                    
        # --- CALIBRACIÓN (RECUPERADA) ---
        with st.expander("📢 Reportar Fallo de la IA"):
            r = st.selectbox("Motivo:", ["Respuesta Obvia", "Alucinación (Inventó ley)", "Pregunta sin sentido"])
            if st.button("Enviar Reporte"):
                engine.feedback_history.append("respuesta_obvia" if "Obvia" in r else "alucinacion")
                st.toast("Ajuste aplicado.")
                
    except Exception as e:
        st.error("Error al mostrar la pregunta. Presiona 'Siguiente' o recarga.")
        if st.button("Resetear Caso"): st.session_state.current_data = None; st.rerun()