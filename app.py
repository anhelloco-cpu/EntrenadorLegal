import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
import requests
from collections import Counter

# --- LIBRERÍAS DEEP LEARNING ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# --- CONFIGURACIÓN VISUAL (ESTÉTICA PROFESIONAL) ---
st.set_page_config(page_title="TITÁN v22 - Edición De Lujo", page_icon="🏛️", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s; background-color: #0d47a1; color: white;}
    .narrative-box {
        background-color: #e3f2fd; padding: 25px; border-radius: 12px; 
        border-left: 6px solid #1565c0; margin-bottom: 25px;
        font-family: 'Georgia', serif; font-size: 1.15em; line-height: 1.6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .question-card {background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; margin-top: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    if DL_AVAILABLE: return SentenceTransformer('all-MiniLM-L6-v2')
    return None

dl_model = load_embedding_model()

# --- LISTADO DE ENTIDADES (NOMBRES OFICIALES COMPLETOS) ---
ENTIDADES_CO = [
    "Contraloría General de la República", 
    "Fiscalía General de la Nación",
    "Procuraduría General de la Nación", 
    "Defensoría del Pueblo",
    "Dirección de Impuestos y Aduanas Nacionales (DIAN)", 
    "Registraduría Nacional del Estado Civil", 
    "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia", 
    "Consejo de Estado", 
    "Corte Constitucional",
    "Policía Nacional de Colombia", 
    "Ejército Nacional de Colombia", 
    "Instituto Colombiano de Bienestar Familiar (ICBF)", 
    "Servicio Nacional de Aprendizaje (SENA)", 
    "Ministerio de Educación Nacional", 
    "Ministerio de Salud y Protección Social", 
    "Departamento Administrativo Nacional de Estadística (DANE)",
    "Otra (Manual) / Agregar +"
]

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
        self.simulacro_mode = False
        self.provider = "Unknown" 
        self.api_key = ""
        self.model = None 
        self.current_temperature = 0.2
        self.last_failed_embedding = None 

    def configure_api(self, key):
        key = key.strip()
        self.api_key = key
        if key.startswith("gsk_"):
            self.provider = "Groq"
            return True, "🚀 Motor GROQ (Llama 3.3) Activado"
        else:
            self.provider = "Google"
            try:
                genai.configure(api_key=key)
                model_list = genai.list_models()
                models = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
                target = next((m for m in models if 'gemini-1.5-pro' in m), 
                         next((m for m in models if 'flash' in m), models[0]))
                self.model = genai.GenerativeModel(target)
                return True, f"🧠 Motor GOOGLE ({target}) Activado"
            except Exception as e:
                return False, f"Error: {str(e)}"

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = [text[i:i+5500] for i in range(0, len(text), 5500)]
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            self.feedback_history = []
            if dl_model: 
                with st.spinner("🧠 Procesando norma..."): self.chunk_embeddings = dl_model.encode(self.chunks)
        else:
            start = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)): self.mastery_tracker[start+i] = 0
            if dl_model: 
                with st.spinner("🧠 Actualizando memoria..."): self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(new_chunks)

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
        score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        perc = int((score / (total * 3)) * 100) if total > 0 else 0
        return min(perc, 100), len(self.failed_indices), total

    # --- REGLAS DE ORO COMPLETAS (SIN RECORTES) ---
    def get_strict_rules(self):
        return """
        🛑 REGLAS DE ORO OBLIGATORIAS (PROTOCOLO ANTI-TEORÍA):
        
        1. PROHIBICIÓN DE "TEORÍA GENERAL":
           - ESTÁ PROHIBIDO preguntar: "¿Qué dice la ley sobre X?". (Esto se responde sin leer).
           - OBLIGATORIO preguntar: "Teniendo en cuenta la conducta del Sr. [Nombre] en la fecha [Fecha], ¿qué norma vulneró?".
           - REGLA DE ORO: Si yo puedo tapar el texto del caso y aún así responder la pregunta, TU TRABAJO ESTÁ MAL HECHO.
        
        2. DEPENDENCIA DE HECHOS (DATA DEPENDENCY):
           - El enunciado de la pregunta DEBE mencionar explícitamente un nombre, una fecha, un cargo o una situación única narrada en el caso.
        
        3. ANTI-SPOILER SEMÁNTICO:
           - No uses palabras en el enunciado que compartan raíz con la respuesta.
           - MALO: "El funcionario omitió..." (Respuesta: Omisión).
           - BUENO: "El funcionario guardó silencio..." (Respuesta: Omisión).
           
        4. TRAMPAS DE COMPETENCIA:
           - Los distractores deben ser leyes reales que no aplican por un detalle técnico.
        """

    # --- MENÚ DE CALIBRACIÓN COMPLETO (SIN RECORTES) ---
    def get_calibration_instructions(self):
        if not self.feedback_history: return ""
        counts = Counter(self.feedback_history)
        instructions = []
        if counts['desconexion'] > 0: instructions.append("🔴 ERROR CRÍTICO: Desconexión temática. ¡ORDEN!: Las preguntas DEBEN basarse 100% en los hechos del caso.")
        if counts['recorte'] > 0: instructions.append("🔴 INTEGRIDAD: ¡PROHIBIDO RESUMIR! Usa los requisitos COMPLETOS de la norma.")
        if counts['spoiler'] > 0: instructions.append("🔴 ALERTA SPOILER: ¡PROHIBIDO incluir la respuesta o pistas obvias en el enunciado!")
        if counts['sesgo_longitud'] > 0: instructions.append("🔴 VISUAL: ¡ALERTA! Las opciones deben tener la misma longitud visual.")
        if counts['respuesta_obvia'] > 0: instructions.append("🔴 DIFICULTAD: Usa 'Trampas de Pertinencia'. Prohibido preguntas que se respondan sin leer el caso.")
        if counts['pregunta_facil'] > 0: instructions.append("🔴 NIVEL EXPERTO: La clave debe ser un detalle minúsculo.")
        if counts['repetitivo'] > 0: self.current_temperature = 0.9; instructions.append("🔴 CREATIVIDAD: ¡CAMBIA TODO!: Nombres, cargos, situaciones.")
        if counts['alucinacion'] > 0: self.current_temperature = 0.0; instructions.append("🔴 FUENTE CERRADA: ¡ESTRICTO! No inventes leyes. Cíñete SOLO al texto.")
        if counts['incoherente'] > 0: instructions.append("🔴 CLARIDAD: Escribe con sintaxis jurídica perfecta.")
        return "\n".join(instructions)

    def generate_case(self):
        if not self.api_key: return {"error": "Falta Llave"}
        if not self.chunks: return {"error": "Falta Norma"}
        
        idx = -1
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]
        
        if idx == -1: idx = random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        
        prompt = f"""
        ACTÚA COMO EXPERTO CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO: {self.entity.upper()}.
        NORMA BASE: "{self.chunks[idx][:6000]}"
        
        {self.get_strict_rules()}
        {self.get_calibration_instructions()}
        
        TAREA:
        1. Redacta un CASO SITUACIONAL complejo y detallado.
        2. Genera 4 PREGUNTAS difíciles y dependientes del texto.
        
        JSON OBLIGATORIO:
        {{
            "narrativa_caso": "Texto...",
            "preguntas": [
                {{
                    "enunciado": "...", 
                    "opciones": {{"A": "..", "B": "..", "C": ".."}}, 
                    "respuesta": "A", 
                    "explicacion": "..."
                }}
            ]
        }}
        """
        
        max_retries = 3
        attempts = 0
        while attempts < max_retries:
            try:
                if self.provider == "Google":
                    safety = [{"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_NONE"} for c in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]]
                    res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "temperature": self.current_temperature}, safety_settings=safety)
                    text_resp = res.text.strip()
                else:
                    headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    data = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "system", "content": "JSON ONLY. STRICT RULES."}, {"role": "user", "content": prompt}],
                        "temperature": self.current_temperature,
                        "response_format": {"type": "json_object"}
                    }
                    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                    text_resp = resp.json()['choices'][0]['message']['content']

                if "```" in text_resp:
                    match = re.search(r'```(?:json)?(.*?)```', text_resp, re.DOTALL)
                    if match: text_resp = match.group(1).strip()
                return json.loads(text_resp)
            except Exception as e:
                time.sleep(5); attempts += 1
        return {"error": "Saturado."}

# --- INTERFAZ ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

with st.sidebar:
    st.title("⚙️ TITÁN v22")
    with st.expander("🔑 LLAVE MAESTRA", expanded=True):
        key = st.text_input("API Key:", type="password")
        if key:
            ok, msg = engine.configure_api(key)
            if ok: st.success(msg)
            else: st.error(msg)
    
    st.divider()
    with st.expander("📂 Cargar Avance", expanded=True):
        upl = st.file_uploader("Archivo:", type=['json'])
        if upl:
            d = json.load(upl)
            engine.chunks = d['chunks']
            engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
            engine.failed_indices = set(d['failed'])
            engine.feedback_history = d.get('feed', [])
            engine.entity = d.get('ent', "")
            st.success("¡Cargado!")
            if engine.api_key: time.sleep(0.5); st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()

    if engine.chunks and engine.api_key and st.session_state.page == 'setup':
        st.divider()
        if st.button("▶️ CONTINUAR", type="primary"): st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()

    st.divider()
    engine.level = st.selectbox("Nivel:", ["Profesional", "Asesor"], index=0)
    
    # Selector de Entidad con Lógica Manual
    ent_selection = st.selectbox("Entidad:", ENTIDADES_CO)
    if "Otra" in ent_selection or "Agregar" in ent_selection:
        engine.entity = st.text_input("Escribe el nombre de la Entidad:")
    else:
        engine.entity = ent_selection

    txt = st.text_area("Cargar Nueva Norma:", height=150)
    if st.button("🚀 INICIAR"):
        if engine.process_law(txt): st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
            
    if st.button("🔥 SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True; st.session_state.current_data = None; st.session_state.page = 'game'; st.rerun()
    
    if engine.chunks:
        save = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "feed": engine.feedback_history, "ent": engine.entity})
        st.download_button("Guardar Progreso", save, "progreso_titan.json")

# --- JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"**DOMINIO: {perc}%** | **BLOQUES: {total}**")
    st.progress(perc/100)

    if not st.session_state.get('current_data'):
        with st.spinner(f"🧠 {engine.provider} analizando..."):
            data = engine.generate_case()
            if data and "preguntas" in data:
                st.session_state.current_data = data
                st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()
            else:
                st.error("Error generación"); st.button("Reintentar", on_click=st.rerun)
                st.stop()

    data = st.session_state.current_data
    st.markdown(f"<div class='narrative-box'><h4>🏛️ {engine.entity}</h4>{data.get('narrativa_caso','Error')}</div>", unsafe_allow_html=True)
    
    q_list = data.get('preguntas', [])
    if q_list:
        q = q_list[st.session_state.q_idx]
        st.write(f"### Pregunta {st.session_state.q_idx + 1}")
        
        with st.form(key=f"q_{st.session_state.q_idx}"):
            sel = st.radio(q['enunciado'], [f"{k}) {v}" for k,v in q['opciones'].items()])
            if st.form_submit_button("Validar"):
                if sel[0] == q['respuesta']: st.success("✅ ¡Correcto!"); engine.mastery_tracker[engine.current_chunk_idx] += 1
                else: st.error(f"Incorrecto. Era {q['respuesta']}"); engine.failed_indices.add(engine.current_chunk_idx)
                st.info(q['explicacion']); st.session_state.answered = True

        if st.session_state.answered:
            if st.session_state.q_idx < len(q_list) - 1:
                if st.button("Siguiente"): st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
            else:
                if st.button("Nuevo Caso"): st.session_state.current_data = None; st.rerun()
        
        # --- MENÚ DE CALIBRACIÓN COMPLETO (RESTAURADO) ---
        st.divider()
        with st.expander("🛠️ CALIBRACIÓN MANUAL (SI FALLA)", expanded=True):
            reasons_map = {
                "Preguntas no tienen que ver con el Caso": "desconexion",
                "Respuesta Incompleta (Recortó la norma)": "recorte",
                "Spoiler (Regala dato)": "spoiler",
                "Respuesta Obvia (Sin leer el caso)": "respuesta_obvia",
                "Alucinación (Inventó ley)": "alucinacion",
                "Opciones Desiguales (Largo)": "sesgo_longitud",
                "Muy Fácil (Dato regalado)": "pregunta_facil",
                "Repetitivo / Poca creatividad": "repetitivo",
                "Incoherente / Mal redactado": "incoherente"
            }
            r = st.selectbox("¿Qué estuvo mal?", list(reasons_map.keys()))
            if st.button("¡Castigar y Corregir!"):
                code = reasons_map[r]
                engine.feedback_history.append(code)
                st.toast(f"Calibración enviada: {code}", icon="🛡️")