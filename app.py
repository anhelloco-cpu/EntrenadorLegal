import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
import requests
from collections import Counter

# --- GESTIÓN DE DEPENDENCIAS (PROFESIONAL) ---
# Intentamos cargar las librerías avanzadas. Si no están, la App NO se rompe,
# simplemente deshabilita esas funciones específicas.

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# --- CONFIGURACIÓN VISUAL (ESTÉTICA DE LUJO) ---
st.set_page_config(page_title="TITÁN v31 - Web App Limpia", page_icon="✨", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s; background-color: #000000; color: white;}
    .narrative-box {
        background-color: #eceff1; padding: 25px; border-radius: 12px; 
        border-left: 6px solid #263238; margin-bottom: 25px;
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

# --- LISTADO DE ENTIDADES (COMPLETO) ---
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

    def extract_methodology_from_pdf(self, pdf_file):
        if not PDF_AVAILABLE: return "Error: Librería PDF no disponible."
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            # Leemos primeras 20 páginas para no saturar
            for i in range(min(len(reader.pages), 20)):
                text += reader.pages[i].extract_text()
            
            prompt = f"""
            ACTÚA COMO UN EXPERTO EN PSICOMETRÍA Y CONCURSOS CNSC.
            Analiza el siguiente texto extraído de una Guía de Orientación:
            TEXTO: "{text[:25000]}"
            TAREA: Extrae y resume en un solo párrafo la METODOLOGÍA DE EVALUACIÓN.
            Busca: ¿Cómo son las preguntas? ¿Cuántas opciones? ¿Qué competencia evalúa?
            Responde SOLO con el párrafo resumen.
            """
            
            if self.provider == "Google":
                res = self.model.generate_content(prompt)
                return res.text
            else:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                data = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]}
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error leyendo PDF: {str(e)}"

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

    # --- REGLAS DE ORO (MUDEZ + ALEATORIEDAD + FORMALIDAD) ---
    def get_strict_rules(self):
        return """
        🛑 PROTOCOLO DE MUDEZ SELECTIVA Y ALEATORIEDAD:
        
        1. ESTRUCTURA DE LA PREGUNTA:
           - La pregunta DEBE ser: [Referencia al Sujeto] + [Referencia a Fecha/Documento] + [Interrogante Jurídico].
           - PROHIBIDO: Usar frases explicativas intermedias que describan la acción o la conducta.
        
        2. DEPENDENCIA TOTAL:
           - El usuario NO DEBE saber qué pasó en esa fecha si no lee el texto.
           
        3. ALEATORIEDAD DE RESPUESTAS (OBLIGATORIO):
           - La respuesta correcta NO DEBE SER SIEMPRE LA "A".
           - Debes distribuir aleatoriamente la respuesta correcta entre las opciones A, B y C.
        """

    def get_calibration_instructions(self):
        if not self.feedback_history: return ""
        counts = Counter(self.feedback_history)
        instructions = []
        if counts['desconexion'] > 0: instructions.append("🔴 ERROR CRÍTICO: Desconexión temática. ¡ORDEN!: Las preguntas DEBEN basarse 100% en los hechos del caso.")
        if counts['recorte'] > 0: instructions.append("🔴 INTEGRIDAD: ¡PROHIBIDO RESUMIR! Usa los requisitos COMPLETOS de la norma.")
        if counts['spoiler'] > 0: instructions.append("🔴 ALERTA SPOILER: ¡PROHIBIDO describir la conducta en la pregunta! Solo usa fechas/nombres.")
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
        
        contexto_adicional = ""
        if self.job_functions:
            contexto_adicional += f"\nCONTEXTO DEL CARGO: {self.job_functions}.\n"
        if self.guide_methodology:
            contexto_adicional += f"\nMETODOLOGÍA OBLIGATORIA (GUÍA): Aplica estrictamente: '{self.guide_methodology}'.\n"
        
        contexto_adicional += f"\nEJE TEMÁTICO: {self.thematic_axis.upper()}\n"

        prompt = f"""
        ACTÚA COMO EXPERTO CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO: {self.entity.upper()}.
        
        {contexto_adicional}
        
        NORMA BASE: "{self.chunks[idx][:6000]}"
        
        {self.get_strict_rules()}
        {self.get_calibration_instructions()}
        
        TAREA:
        1. Redacta un CASO SITUACIONAL complejo (Fechas, Nombres, Documentos).
        2. Genera 4 PREGUNTAS (Estilo Críptico/Mudo y Aleatorio).
        
        FORMATO DE EXPLICACIÓN OBLIGATORIO (ESTRICTO):
        En el campo "explicacion", DEBES seguir esta estructura exacta:
        "NORMA TAXATIVA: [Cita textual entre comillas] ... ANÁLISIS: [Explicación de por qué aplica al caso] ... DESCARTES: [Por qué las otras opciones no aplican]"
        
        JSON OBLIGATORIO:
        {{
            "narrativa_caso": "Texto...",
            "preguntas": [
                {{
                    "enunciado": "...", 
                    "opciones": {{"A": "..", "B": "..", "C": ".."}}, 
                    "respuesta": "A", 
                    "explicacion": "NORMA TAXATIVA: ... ANÁLISIS: ... DESCARTES: ..."
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
    st.title("⚙️ TITÁN v31")
    with st.expander("🔑 LLAVE MAESTRA", expanded=True):
        key = st.text_input("API Key:", type="password")
        if key:
            ok, msg = engine.configure_api(key)
            if ok: st.success(msg)
            else: st.error(msg)
    
    st.divider()
    
    # --- PANEL DE ESTRATEGIA (INTELIGENTE) ---
    st.markdown("### 📋 ESTRATEGIA DE ESTUDIO")
    with st.expander("1. Configurar Contexto", expanded=False):
        engine.job_functions = st.text_area("Funciones del Cargo:", placeholder="Ej: Atención al ciudadano...", height=80)
        
        # LÓGICA CONDICIONAL: Solo muestra el uploader si PyPDF2 está instalado en el servidor
        if PDF_AVAILABLE:
            uploaded_pdf = st.file_uploader("Subir Guía de Orientación (PDF)", type="pdf")
            if uploaded_pdf is not None:
                if st.button("🔍 Extraer Metodología del PDF"):
                    with st.spinner("Leyendo y analizando Guía..."):
                        extracted = engine.extract_methodology_from_pdf(uploaded_pdf)
                        engine.guide_methodology = extracted
                        st.success("¡Metodología Extraída!")
        else:
            st.warning("⚠️ Módulo PDF no detectado (PyPDF2). Pega la guía manualmente:")
            
        # Campo editable (siempre visible para correcciones manuales o pegado directo)
        engine.guide_methodology = st.text_area("Metodología Activa:", value=engine.guide_methodology, height=150)

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
    axis_input = st.text_input("Nombre Eje Temático:", value="General")
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
        # Mensaje de carga
        loading_msg = f"🧠 {engine.provider} analizando..."
        if engine.guide_methodology: loading_msg = "🧠 Aplicando Metodología de la Guía..."
        
        with st.spinner(loading_msg):
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
        
        # --- CALIBRACIÓN MANUAL (INTACTA) ---
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