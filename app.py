import streamlit as st
import google.generativeai as genai
import json
import random
import time
import requests
from collections import Counter

# --- GESTIÓN DE DEPENDENCIAS ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="TITÁN v37 - Estructura Total", page_icon="🏛️", layout="wide")
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

# --- LISTADO DE ENTIDADES ---
ENTIDADES_CO = [
    "Contraloría General de la República", "Fiscalía General de la Nación",
    "Procuraduría General de la Nación", "Defensoría del Pueblo",
    "DIAN", "Registraduría Nacional", "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia", "Consejo de Estado", "Corte Constitucional",
    "Policía Nacional", "Ejército Nacional", "ICBF", "SENA", 
    "Ministerio de Educación", "Ministerio de Salud", "DANE",
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
        # --- VARIABLES DE ESTRATEGIA (RESTAURADAS) ---
        self.study_phase = "Pre-Guía" # Pre-Guía o Post-Guía
        self.job_functions = ""
        self.guide_methodology = ""
        self.thematic_axis = "General"

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

    def process_law(self, text, axis_name, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        self.thematic_axis = axis_name 
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

    # --- REGLAS DE ORO ---
    def get_strict_rules(self):
        return """
        🛑 REGLAS DE ORO OBLIGATORIAS:
        1. NO SPOILERS: La pregunta NO debe describir la conducta ilegal ni dar la respuesta.
           - Malo: "Dado que el funcionario robó..."
           - Bueno: "Frente a la actuación descrita en el párrafo 2..."
        2. DEPENDENCIA: El usuario debe estar obligado a leer el texto para responder.
        3. ALEATORIEDAD: La respuesta correcta NO puede ser siempre la A. Distribúyela.
        """

    # --- CALIBRACIÓN ---
    def get_calibration_instructions(self):
        if not self.feedback_history: return ""
        counts = Counter(self.feedback_history)
        instructions = []
        if counts['desconexion'] > 0: instructions.append("🔴 ERROR: Desconexión temática. ¡Cíñete al caso!")
        if counts['recorte'] > 0: instructions.append("🔴 ERROR: Respuesta incompleta. ¡Usa la norma taxativa!")
        if counts['spoiler'] > 0: instructions.append("🔴 ERROR: Spoiler. ¡No describas la conducta en la pregunta!")
        if counts['respuesta_obvia'] > 0: instructions.append("🔴 ERROR: Muy obvio. ¡Sube la dificultad!")
        if counts['alucinacion'] > 0: instructions.append("🔴 ERROR: Alucinación. ¡Solo usa la ley provista!")
        if counts['sesgo_longitud'] > 0: instructions.append("🔴 ERROR: Opciones desiguales. ¡Equilibra la longitud!")
        if counts['pregunta_facil'] > 0: instructions.append("🔴 ERROR: Demasiado fácil. ¡Pon trampas!")
        if counts['repetitivo'] > 0: self.current_temperature = 0.9; instructions.append("🔴 ERROR: Repetitivo. ¡Sé más creativo!")
        if counts['incoherente'] > 0: instructions.append("🔴 ERROR: Redacción. ¡Mejora la sintaxis!")
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
        
        # --- CEREBRO ESTRATÉGICO (Pre vs Post) ---
        contexto_estrategia = ""
        
        if self.study_phase == "Pre-Guía":
            # MODALIDAD 1: PRE-GUÍA (Enfoque en Funciones + Norma)
            # Aquí la IA asume un rol de entrenamiento funcional general.
            contexto_estrategia = f"""
            MODO DE ESTUDIO: PRE-GUÍA (Entrenamiento Funcional).
            CONTEXTO: El usuario aún no tiene la guía oficial, así que estudiaremos la relación entre la NORMA y las FUNCIONES DEL CARGO.
            FUNCIONES DEL USUARIO: '{self.job_functions}'
            INSTRUCCIÓN: Crea un caso donde el usuario deba aplicar la norma cargada para resolver un problema relacionado con sus funciones.
            """
        else:
            # MODALIDAD 2: POST-GUÍA (Enfoque Metodológico Estricto)
            # Aquí la IA obedece ciegamente el texto que pegues de la guía.
            contexto_estrategia = f"""
            MODO DE ESTUDIO: POST-GUÍA (Simulacro Oficial).
            CONTEXTO: El usuario tiene la Guía de Orientación en mano.
            METODOLOGÍA OBLIGATORIA (TEXTO DE LA GUÍA):
            '''{self.guide_methodology}'''
            
            INSTRUCCIÓN SUPREMA: 
            1. Analiza el texto de la guía proporcionado arriba.
            2. Extrae el estilo de pregunta (¿Juicio Situacional? ¿Análisis de Caso? ¿Contexto Técnico?).
            3. Extrae el número de opciones (¿Son 3 o 4?).
            4. Genera el caso y las preguntas SIGUIENDO EXACTAMENTE ESA METODOLOGÍA.
            """

        prompt = f"""
        ACTÚA COMO EXPERTO EN CONCURSOS PÚBLICOS (NIVEL {self.level.upper()}).
        ENTIDAD: {self.entity.upper()}. EJE TEMÁTICO: {self.thematic_axis.upper()}.
        
        {contexto_estrategia}
        
        NORMA BASE: "{self.chunks[idx][:7000]}"
        
        {self.get_strict_rules()}
        {self.get_calibration_instructions()}
        
        TAREA:
        1. Redacta el Caso/Contexto (Adaptado a la fase de estudio).
        2. Genera 4 preguntas (O las que dicte la guía en modo Post).
        
        FORMATO DE RESPUESTA OBLIGATORIO:
        En "explicacion", usa: "NORMA TAXATIVA: ... ANÁLISIS: ... DESCARTES: ..."
        
        JSON OBLIGATORIO:
        {{
            "narrativa_caso": "...",
            "preguntas": [
                {{
                    "enunciado": "...", 
                    "opciones": {{"A": "...", "B": "...", "C": "..."}}, 
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
                        "messages": [{"role": "system", "content": "JSON ONLY."}, {"role": "user", "content": prompt}],
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
                time.sleep(2); attempts += 1
        return {"error": "Saturado."}

# --- INTERFAZ ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

with st.sidebar:
    st.title("⚙️ TITÁN v37 (Restaurado)")
    with st.expander("🔑 LLAVE MAESTRA", expanded=True):
        key = st.text_input("API Key:", type="password")
        if key:
            ok, msg = engine.configure_api(key)
            if ok: st.success(msg)
            else: st.error(msg)
    
    st.divider()
    
    # --- PANEL DE ESTRATEGIA (CON SELECTOR DE FASE) ---
    st.markdown("### 📋 ESTRATEGIA DE ESTUDIO")
    
    # 1. EL SELECTOR QUE QUERÍAS
    fase = st.radio("Fase de Preparación:", ["Pre-Guía", "Post-Guía"], index=0, help="Pre: Estudia funciones. Post: Aplica la guía del concurso.")
    engine.study_phase = fase

    # 2. LOS CAMPOS SEGÚN LA FASE
    with st.expander("Configurar Contexto", expanded=True):
        if fase == "Pre-Guía":
            st.info("📌 Modo Funcional: Entrena tus funciones con la norma.")
            engine.job_functions = st.text_area("Tus Funciones:", height=100, placeholder="Ej: Atender peticiones, Sustanciar fallos...")
            engine.guide_methodology = "" # Se limpia para no confundir
        else:
            st.warning("📌 Modo Metodológico: La IA obedecerá tu Guía.")
            engine.guide_methodology = st.text_area("📜 PEGA AQUÍ LA METODOLOGÍA (De la Guía PDF):", height=150, placeholder="Ej: 'Preguntas de Análisis, 4 opciones de respuesta, contexto técnico...'")
            engine.job_functions = "" # Se limpia

    st.divider()
    
    with st.expander("2. Cargar Normas", expanded=True):
        upl = st.file_uploader("Cargar Backup JSON:", type=['json'])
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
        if st.button("▶️ IR AL SIMULACRO", type="primary"): st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()

    st.divider()
    engine.level = st.selectbox("Nivel:", ["Profesional", "Asesor", "Técnico", "Asistencial"], index=0)
    
    ent_selection = st.selectbox("Entidad:", ENTIDADES_CO)
    if "Otra" in ent_selection or "Agregar" in ent_selection:
        engine.entity = st.text_input("Nombre Entidad:")
    else:
        engine.entity = ent_selection

    st.markdown("---")
    axis_input = st.text_input("Eje Temático:", value="General")
    txt = st.text_area("📜 Pegar Norma:", height=150)
    
    if st.button("🚀 PROCESAR NORMA"):
        if engine.process_law(txt, axis_input): st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
            
    if st.button("🔥 INICIAR SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True; st.session_state.current_data = None; st.session_state.page = 'game'; st.rerun()
    
    if engine.chunks:
        save = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "feed": engine.feedback_history, "ent": engine.entity})
        st.download_button("💾 Guardar Progreso", save, "progreso_titan.json")

# --- JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    st.markdown(f"**EJE: {engine.thematic_axis.upper()}** | **DOMINIO: {perc}%** | **BLOQUES: {total}**")
    st.progress(perc/100)

    if not st.session_state.get('current_data'):
        # Mensaje de carga inteligente
        msg = "🧠 Analizando norma..."
        if engine.study_phase == "Post-Guía": msg = "🧠 Leyendo tu Guía y adaptando metodología..."
        
        with st.spinner(msg):
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
            # Filtro inteligente de opciones vacías (Por si la guía pide 3)
            opciones_validas = {k: v for k, v in q['opciones'].items() if v}
            sel = st.radio(q['enunciado'], [f"{k}) {v}" for k,v in opciones_validas.items()])
            
            if st.form_submit_button("Validar"):
                letra_sel = sel.split(")")[0]
                if letra_sel == q['respuesta']: st.success("✅ ¡Correcto!"); engine.mastery_tracker[engine.current_chunk_idx] += 1
                else: st.error(f"Incorrecto. Era {q['respuesta']}"); engine.failed_indices.add(engine.current_chunk_idx)
                st.info(q['explicacion']); st.session_state.answered = True

        if st.session_state.answered:
            if st.session_state.q_idx < len(q_list) - 1:
                if st.button("Siguiente"): st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
            else:
                if st.button("Nuevo Caso"): st.session_state.current_data = None; st.rerun()
        
        # --- CALIBRACIÓN COMPLETA (INTACTA) ---
        st.divider()
        with st.expander("🛠️ CALIBRACIÓN MANUAL", expanded=True):
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