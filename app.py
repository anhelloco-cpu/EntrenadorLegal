import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- LIBRERÍAS DEEP LEARNING (Opcionales) ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="TITÁN v8.5 - Stable", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s;}
    .narrative-box {
        background-color: #e0f7fa; padding: 25px; border-radius: 12px; 
        border-left: 6px solid #006064; margin-bottom: 25px;
        font-family: 'Georgia', serif; font-size: 1.15em;
    }
    .question-card {background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    if DL_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

dl_model = load_embedding_model()

# --- ENTIDADES COLOMBIA ---
ENTIDADES_CO = [
    "Contraloría General de la República", "Fiscalía General de la Nación",
    "Procuraduría General de la Nación", "Defensoría del Pueblo",
    "DIAN", "Registraduría Nacional", "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia", "Consejo de Estado", "Corte Constitucional",
    "Policía Nacional", "Ejército Nacional", "ICBF", "SENA", 
    "Ministerio de Educación", "Ministerio de Salud", "DANE",
    "Otra (Manual) / Agregar +"
]

# --- 2. MOTOR LÓGICO ---
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
        self.last_error = ""

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            model_list = genai.list_models()
            available_models = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
            if not available_models: return False, "API Key sin acceso a modelos."

            target_model = available_models[0]
            for m in available_models:
                if 'flash' in m.lower(): target_model = m; break
                if 'pro' in m.lower() and 'vision' not in m.lower(): target_model = m 
            
            self.model = genai.GenerativeModel(target_model)
            return True, f"Conectado a: {target_model}"
        except Exception as e: return False, str(e)

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = [text[i:i+5500] for i in range(0, len(text), 5500)]
        
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            self.mistakes_log = []
            self.feedback_history = [] 
            if dl_model: 
                with st.spinner("🧠 Procesando norma..."):
                    self.chunk_embeddings = dl_model.encode(self.chunks)
        else:
            start = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)): self.mastery_tracker[start+i] = 0
            if dl_model: 
                with st.spinner("🧠 Actualizando memoria..."):
                    self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(new_chunks)

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
        score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        perc = int((score / (total * 3)) * 100) if total > 0 else 0
        return min(perc, 100), len(self.failed_indices), total

    def get_calibration_prompt(self):
        if not self.feedback_history: return "Modo: Estándar (Sin ajustes previos)."
        counts = Counter(self.feedback_history)
        instructions = []
        
        if counts['recorte'] > 0: instructions.append("⚠️ INTEGRIDAD CRÍTICA: Has sido reportado por recortar la norma. Debes incluir TODOS los requisitos (A, B y C). Prohibido resumir.")
        if counts['spoiler'] > 0: instructions.append("🔗 ANTI-SPOILER EXTREMO: El enunciado NO puede contener la respuesta. El usuario debe deducirlo de la narrativa.")
        if counts['sesgo_longitud'] > 0: instructions.append("🛑 FORMATO: Las opciones deben tener la misma longitud visual (palabras) para no delatar la correcta.")
        if counts['respuesta_obvia'] > 0: instructions.append("💀 DIFICULTAD: Los distractores son muy obvios. Deben ser 'Trampas de Pertinencia' (Leyes reales que parecen aplicar pero no).")
        if counts['pregunta_facil'] > 0: instructions.append("🔍 DETALLE: La clave de la respuesta debe ser un detalle minúsculo (un plazo, una excepción, una autoridad).")
        if counts['repetitivo'] > 0:
            self.current_temperature = 0.8; instructions.append("🔄 CREATIVIDAD: Cambia radicalmente los nombres, los cargos y el tipo de problema jurídico.")
        if counts['alucinacion'] > 0:
            self.current_temperature = 0.0; instructions.append("⛔ FUENTE CERRADA: Prohibido inventar leyes. Usa SOLO el texto provisto.")
        if counts['incoherente'] > 0: instructions.append("🧠 LÓGICA: La redacción anterior fue confusa. Escribe con claridad jurídica perfecta.")

        return "\n".join(instructions)

    def generate_case(self):
        if not self.model: return {"error": "⚠️ Primero conecta tu API Key en el menú lateral."}
        if not self.chunks: return {"error": "Carga una norma primero."}
        
        idx = -1
        selection_reason = "Aleatorio"

        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]; selection_reason = "Deep Learning"
        
        if idx == -1:
            if self.simulacro_mode: idx = random.choice(range(len(self.chunks)))
            elif self.failed_indices and random.random() < 0.6: idx = random.choice(list(self.failed_indices)); selection_reason = "Repaso"
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        self.selection_reason = selection_reason
        
        instruccion_nivel = ""
        if self.level in ["Profesional", "Asesor"]:
            instruccion_nivel = """
            NIVEL EXPERTO (HARDCORE):
            - TODAS las opciones (A, B, C) deben ser VERDADERAS jurídicamente.
            - SOLO UNA aplica a este caso por un detalle de competencia o procedimiento.
            - Las otras son errores de subsunción (ley correcta, caso incorrecto).
            """
        
        prompt = f"""
        ACTÚA COMO EXPERTO CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO: {self.entity.upper()}.
        NORMA: "{self.chunks[idx][:6000]}"
        
        {instruccion_nivel}
        
        TAREA:
        1. Caso complejo en {self.entity}.
        2. 4 PREGUNTAS difíciles.
        
        REGLAS DE RETROALIMENTACIÓN:
        En 'explicacion' DEBES estructurar así:
        - "NORMA TAXATIVA": Cita textual.
        - "ANÁLISIS": Por qué aplica.
        - "DESCARTES": Por qué las otras no aplican (aunque sean leyes reales).
        
        !!! AJUSTES ACTIVOS !!!:
        {self.get_calibration_prompt()}
        
        JSON OBLIGATORIO:
        {{
            "narrativa_caso": "Historia...",
            "preguntas": [
                {{"enunciado": "...", "opciones": {{"A": "..", "B": "..", "C": ".."}}, "respuesta": "A", "explicacion": "NORMA TAXATIVA: ... ANÁLISIS: ..."}},
                {{"enunciado": "...", "opciones": {{"A": "..", "B": "..", "C": ".."}}, "respuesta": "B", "explicacion": "NORMA TAXATIVA: ... ANÁLISIS: ..."}},
                {{"enunciado": "...", "opciones": {{"A": "..", "B": "..", "C": ".."}}, "respuesta": "C", "explicacion": "NORMA TAXATIVA: ... ANÁLISIS: ..."}},
                {{"enunciado": "...", "opciones": {{"A": "..", "B": "..", "C": ".."}}, "respuesta": "A", "explicacion": "NORMA TAXATIVA: ... ANÁLISIS: ..."}}
            ]
        }}
        """
        try:
            res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "temperature": self.current_temperature})
            text_resp = res.text.strip()
            if "```" in text_resp:
                match = re.search(r'```(?:json)?(.*?)```', text_resp, re.DOTALL)
                if match: text_resp = match.group(1).strip()
            return json.loads(text_resp)
        except Exception as e:
            self.last_error = str(e)
            return None

# --- 3. INTERFAZ ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

with st.sidebar:
    st.title("⚙️ TITÁN v8.5")
    if DL_AVAILABLE: st.success("🧠 Neurona: ACTIVADA")
    
    key = st.text_input("1. API Key (Obligatorio):", type="password")
    if key and not engine.model:
        ok, msg = engine.configure_api(key)
        if ok: st.success(msg)
        else: st.error(msg)
    
    st.divider()

    with st.expander("2. 📂 Cargar Avance (JSON)", expanded=True):
        upl = st.file_uploader("Archivo:", type=['json'])
        if upl:
            try:
                d = json.load(upl)
                engine.chunks = d['chunks']
                engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
                engine.failed_indices = set(d['failed'])
                engine.feedback_history = d.get('feed', [])
                engine.entity = d.get('ent', "")
                
                if engine.model:
                    st.success("¡Datos y Conexión listos! Iniciando...")
                    time.sleep(1)
                    st.session_state.page = 'game'
                    st.session_state.current_data = None
                    st.rerun()
                else:
                    st.warning("✅ Datos cargados. AHORA INGRESA TU API KEY ARRIBA para iniciar.")
            except: st.error("Archivo inválido")

    st.divider()
    
    engine.level = st.selectbox("Nivel:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    ent_sel = st.selectbox("Entidad:", ENTIDADES_CO)
    if "Otra" in ent_sel or "Agregar" in ent_sel: engine.entity = st.text_input("Nombre Entidad:")
    else: engine.entity = ent_sel

    txt = st.text_area("Cargar Nueva Norma:", height=150)
    col1, col2 = st.columns(2)
    if col1.button("🚀 INICIAR"):
        if engine.process_law(txt): st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
    if col2.button("➕ SUMAR"):
        if engine.process_law(txt, True): st.success("Agregado.")
            
    st.divider()
    if st.button("🔥 SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True; st.session_state.current_data = None; st.session_state.page = 'game'; st.rerun()

    if engine.chunks:
        save = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "feed": engine.feedback_history, "ent": engine.entity})
        st.download_button("Guardar Progreso", save, "progreso_titan.json")

# --- 4. JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"**DOMINIO: {perc}%** | **BLOQUES: {total}** | **REPASOS: {fails}**")
    st.progress(perc/100)

    if not st.session_state.get('current_data'):
        msg = "🧠 Generando caso..."
        if DL_AVAILABLE and engine.last_failed_embedding is not None: msg = "🧠 Neurona atacando debilidad..."
        
        with st.spinner(msg):
            data = engine.generate_case()
            # Validación robusta
            if data and isinstance(data, dict) and "preguntas" in data and len(data['preguntas']) > 0:
                st.session_state.current_data = data
                st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()
            else:
                # FIX: Manejo seguro del error sin crashear
                error_txt = "Error desconocido"
                if isinstance(data, dict): error_txt = data.get('error', engine.last_error)
                else: error_txt = engine.last_error if engine.last_error else "Respuesta vacía de la IA"
                
                st.error(f"⚠️ {error_txt}")
                if "API Key" in str(error_txt): st.info("Ve al menú lateral y conecta tu llave.")
                elif st.button("🔄 REINTENTAR"): st.rerun()
                st.stop()

    data = st.session_state.current_data
    st.markdown(f"<div class='narrative-box'><h4>🏛️ {engine.entity}</h4>{data.get('narrativa_caso','Error')}</div>", unsafe_allow_html=True)
    
    try:
        q_list = data['preguntas']
        if st.session_state.q_idx >= len(q_list): st.session_state.q_idx = 0

        q = q_list[st.session_state.q_idx]
        st.write(f"### Pregunta {st.session_state.q_idx + 1} de {len(q_list)}")
        
        with st.form(key=f"q_{st.session_state.q_idx}"):
            sel = st.radio(q['enunciado'], [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
            if st.form_submit_button("Validar"):
                if sel and sel[0] == q['respuesta']:
                    st.success("✅ ¡Correcto!"); engine.last_failed_embedding = None
                    if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                else:
                    st.error(f"Incorrecto. Era {q['respuesta']}"); engine.failed_indices.add(engine.current_chunk_idx)
                    if DL_AVAILABLE and engine.chunk_embeddings is not None:
                        engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                st.info(q['explicacion']); st.session_state.answered = True

        if st.session_state.answered:
            if st.session_state.q_idx < len(q_list) - 1:
                if st.button("Siguiente Pregunta ⏭️"):
                    st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
            else:
                if st.button("🔄 Finalizar Caso (Siguiente Bloque)"):
                    engine.mastery_tracker[engine.current_chunk_idx] += 1
                    st.session_state.current_data = None; st.rerun()

        with st.expander("📢 Reportar Fallo (Calibrar IA)", expanded=True):
            reasons_map = {
                "Respuesta Incompleta (Recortó la norma)": "recorte",
                "Spoiler (Regala dato)": "spoiler",
                "Respuesta Obvia / Tonta": "respuesta_obvia",
                "Alucinación (Inventó ley)": "alucinacion",
                "Opciones Desiguales (Largo)": "sesgo_longitud",
                "Muy Fácil (Dato regalado)": "pregunta_facil",
                "Repetitivo / Poca creatividad": "repetitivo",
                "Incoherente / Mal redactado": "incoherente"
            }
            r = st.selectbox("¿Qué estuvo mal?", list(reasons_map.keys()))
            if st.button("Enviar Reporte y Calibrar"):
                code = reasons_map[r]
                engine.feedback_history.append(code)
                st.toast(f"Calibración aplicada: {code}", icon="🛠️")
                
    except Exception as e:
        st.error(f"Error visual: {str(e)}")
        if st.button("Resetear"): st.session_state.current_data = None; st.rerun()