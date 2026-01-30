import streamlit as st
import google.generativeai as genai
import json
import random
import time
import requests
import re
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
st.set_page_config(page_title="TITÁN v52 - Segmentación Inteligente", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s; background-color: #000000; color: white;}
    .narrative-box {
        background-color: #f5f5f5; padding: 25px; border-radius: 12px; 
        border-left: 6px solid #424242; margin-bottom: 25px;
        font-family: 'Georgia', serif; font-size: 1.15em; line-height: 1.6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    if DL_AVAILABLE: return SentenceTransformer('all-MiniLM-L6-v2')
    return None

dl_model = load_embedding_model()

# --- ENTIDADES ---
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
        
        # --- VARIABLES DE CONTROL ---
        self.study_phase = "Pre-Guía" 
        self.example_question = "" 
        self.job_functions = ""    
        self.thematic_axis = "General"
        self.structure_type = "Técnico / Normativo (Sin Caso)" 
        self.questions_per_case = 1 
        
        # --- NUEVO: ESTRUCTURA SEGMENTADA (TU CÓDIGO) ---
        self.sections_map = {} # Diccionario { "TÍTULO I": "texto...", "CAPÍTULO II": "texto..." }
        self.active_section_name = "Todo el Documento"

    def configure_api(self, key):
        key = key.strip()
        self.api_key = key
        
        if key.startswith("gsk_"):
            self.provider = "Groq"
            return True, "🚀 Motor GROQ Activado"
        elif key.startswith("sk-") and not key.startswith("sk-ant"): 
            self.provider = "OpenAI"
            return True, "🤖 Motor CHATGPT (GPT-4o) Activado"
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
                return False, f"Error con la llave: {str(e)}"

    # --- AQUÍ ESTÁ LA INTEGRACIÓN DE TU CÓDIGO DE COLAB ---
    def smart_segmentation(self, full_text):
        """
        Aplica los patrones Regex de tu Colab para detectar Títulos, Capítulos y Artículos,
        y segmenta el texto en bloques lógicos.
        """
        lineas = full_text.split('\n')
        secciones = {"Todo el Documento": full_text} # Opción por defecto
        
        current_label = "Preámbulo/Inicio"
        current_content = []
        
        # TUS PATRONES EXACTOS
        patron_art = r'^\s*(ARTÍCULO|ARTICULO|ART)\.?\s*\d+'
        patron_cap = r'^\s*(CAPÍTULO|CAPITULO)\b'
        patron_tit_txt = r'^\s*(TÍTULO|TITULO|LIBRO|PARTE)\b'
        patron_romano_punto = r'^\s*([IVXLCDM]+)\.\s+(.+)'
        patron_romano_solo = r'^\s*([IVXLCDM]+)\s*$'

        idx = 0
        while idx < len(lineas):
            linea = lineas[idx].strip()
            if not linea:
                idx += 1
                continue
            
            tipo_detectado = None
            etiqueta_nueva = None

            # 1. ROMANO + PUNTO ("I. DISPOSICIONES")
            match_romano = re.match(patron_romano_punto, linea, re.IGNORECASE)
            if match_romano:
                etiqueta_nueva = f"TÍTULO {match_romano.group(1)}: {match_romano.group(2)[:30]}..."
                tipo_detectado = 'TÍTULO'

            # 2. TÍTULO EXPLÍCITO ("TÍTULO I")
            elif re.match(patron_tit_txt, linea, re.IGNORECASE):
                etiqueta_nueva = linea[:50]
                tipo_detectado = 'TÍTULO'

            # 3. ROMANO SOLO ("I")
            elif re.match(patron_romano_solo, linea, re.IGNORECASE):
                palabra = linea.split()[0].upper().replace('.', '')
                if palabra in ['I','II','III','IV','V','VI','VII','VIII','IX','X','L','C','D','M']:
                    etiqueta_nueva = f"SECCIÓN {palabra}"
                    tipo_detectado = 'SECCIÓN'

            # 4. CAPÍTULO ("CAPÍTULO II")
            elif re.match(patron_cap, linea, re.IGNORECASE):
                etiqueta_nueva = linea[:50]
                tipo_detectado = 'CAPÍTULO'

            # 5. ARTÍCULO (Opcional: Si quieres hilar muy fino, descomenta abajo. 
            # Por ahora agrupamos por Capítulos/Títulos para no tener 300 opciones)
            # elif re.match(patron_art, linea, re.IGNORECASE):
            #     etiqueta_nueva = linea[:20]
            #     tipo_detectado = 'ARTÍCULO'

            if tipo_detectado and etiqueta_nueva:
                # Guardar lo anterior
                if current_content:
                    # Si ya existe la clave, anexamos (raro, pero posible)
                    texto_acumulado = "\n".join(current_content)
                    if current_label in secciones:
                        secciones[current_label] += "\n" + texto_acumulado
                    else:
                        secciones[current_label] = texto_acumulado
                
                # Iniciar nueva sección
                current_label = etiqueta_nueva
                current_content = [linea] # Incluir el título en el contenido
            else:
                current_content.append(linea)
            
            idx += 1
        
        # Guardar el último bloque
        if current_content:
             texto_acumulado = "\n".join(current_content)
             secciones[current_label] = texto_acumulado
             
        return secciones

    def process_law(self, text, axis_name):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        self.thematic_axis = axis_name 
        
        # 1. EJECUTAR TU LÓGICA DE SEGMENTACIÓN
        self.sections_map = self.smart_segmentation(text)
        
        # 2. PREPARAR CHUNKS (Por defecto carga TODO)
        # Si el usuario elige una sección específica, actualizaremos self.chunks en la UI
        self.chunks = [text[i:i+6000] for i in range(0, len(text), 6000)]
        self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
        
        if dl_model: 
            with st.spinner("🧠 Analizando estructura de la norma..."): 
                self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(self.chunks)

    def update_chunks_by_section(self, section_name):
        """Actualiza los chunks para que la IA solo vea la sección elegida"""
        if section_name in self.sections_map:
            texto_seccion = self.sections_map[section_name]
            # Reprocesar solo ese texto
            self.chunks = [texto_seccion[i:i+6000] for i in range(0, len(texto_seccion), 6000)]
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))} # Reset mastery for new view
            self.active_section_name = section_name
            if dl_model:
                 self.chunk_embeddings = dl_model.encode(self.chunks)
            return True
        return False

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
        score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        perc = int((score / (total * 3)) * 100) if total > 0 else 0
        return min(perc, 100), len(self.failed_indices), total

    def get_strict_rules(self):
        return """
        🛑 REGLAS DE ORO DE SEGURIDAD:
        1. NO SPOILERS: La pregunta NO debe describir la conducta ilegal ni dar la respuesta en el enunciado.
        2. DEPENDENCIA: El usuario debe estar obligado a leer el texto para responder.
        3. ALEATORIEDAD: La respuesta correcta NO puede ser siempre la A. Distribúyela.
        """

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
        
        # --- PROMPT ---
        if "Sin Caso" in self.structure_type:
            instruccion_estilo = f"""
            ESTILO: TÉCNICO / NORMATIVO.
            1. COPIA LA SINTAXIS EXACTA DEL EJEMPLO DE USUARIO (Si existe):
               - ¿El ejemplo NO usa signos de interrogación '¿?'? -> TÚ TAMPOCO.
               - ¿El ejemplo termina en dos puntos ':'? -> TÚ TAMBIÉN.
               - ¿Usa conectores como "En ese sentido..."? -> ÚSALOS.
            2. FUSIÓN: Genera un solo bloque de texto continuo (Contexto + Enunciado).
            """
        else:
            instruccion_estilo = f"""
            ESTILO: NARRATIVO / SITUACIONAL.
            1. Crea una historia laboral realista con roles definidos.
            2. Funciones/Rol a usar: '{self.job_functions}'
            """

        cantidad_instruccion = f"Genera EXACTAMENTE {self.questions_per_case} ítem(s)."

        prompt = f"""
        ACTÚA COMO EXPERTO EN CONCURSOS PÚBLICOS (NIVEL {self.level.upper()}).
        ENTIDAD: {self.entity.upper()}. EJE: {self.thematic_axis.upper()}.
        SECCIÓN DE ESTUDIO ACTUAL: {self.active_section_name}
        
        {instruccion_estilo}
        
        CANTIDAD REQUERIDA: {cantidad_instruccion}
        
        EJEMPLO SINTÁCTICO A COPIAR:
        '''{self.example_question}'''
        
        NORMA BASE (TEXTO REAL): "{self.chunks[idx][:7000]}"
        
        {self.get_strict_rules()}
        {self.get_calibration_instructions()}
        
        FORMATO JSON OBLIGATORIO:
        {{
            "narrativa_caso": "Si es estilo TÉCNICO, pon el párrafo de contexto. Si es NARRATIVO, pon la historia.",
            "preguntas": [
                {{
                    "enunciado": "Conector y enunciado final...", 
                    "opciones": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, 
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
                # OPENAI
                if self.provider == "OpenAI":
                    headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    data = {
                        "model": "gpt-4o", 
                        "messages": [
                            {"role": "system", "content": "You are a specialized legal assistant. Output valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.current_temperature,
                        "response_format": {"type": "json_object"}
                    }
                    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
                    if resp.status_code != 200: return {"error": f"Error OpenAI: {resp.text}"}
                    text_resp = resp.json()['choices'][0]['message']['content']

                # GOOGLE
                elif self.provider == "Google":
                    safety = [{"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_NONE"} for c in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]]
                    res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "temperature": self.current_temperature}, safety_settings=safety)
                    text_resp = res.text.strip()
                
                # GROQ
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
    st.title("⚙️ TITÁN v52 (Segmentado)")
    with st.expander("🔑 LLAVE MAESTRA", expanded=True):
        key = st.text_input("API Key (Cualquiera):", type="password")
        if key:
            ok, msg = engine.configure_api(key)
            if ok: st.success(msg)
            else: st.error(msg)
    
    st.divider()
    
    # --- ESTRATEGIA ---
    st.markdown("### 📋 ESTRATEGIA")
    fase_default = 0 if engine.study_phase == "Pre-Guía" else 1
    fase = st.radio("Fase:", ["Pre-Guía", "Post-Guía"], index=fase_default)
    engine.study_phase = fase

    st.markdown("#### 🔧 ESTRUCTURA")
    col1, col2 = st.columns(2)
    with col1:
        idx_struct = 0 if "Sin Caso" in engine.structure_type else 1
        estilo = st.radio("Enunciado:", 
                         ["Técnico / Normativo (Sin Caso)", "Narrativo / Situacional (Con Caso)"], 
                         index=idx_struct)
        engine.structure_type = estilo
    
    with col2:
        cant = st.number_input("Preguntas:", min_value=1, max_value=5, value=engine.questions_per_case)
        engine.questions_per_case = cant

    with st.expander("Detalles", expanded=True):
        if "Con Caso" in estilo:
            engine.job_functions = st.text_area("Funciones / Rol:", value=engine.job_functions, height=70, placeholder="Ej: Profesional Universitario...")
            engine.example_question = ""
        else:
            engine.example_question = st.text_area("Ejemplo de Estilo (Sintaxis):", value=engine.example_question, height=70, placeholder="Pega el ejemplo para copiar los 'dos puntos' y conectores...")
            engine.job_functions = ""

    st.divider()
    
    # --- PESTAÑAS DE CARGA ---
    tab1, tab2 = st.tabs(["📝 NUEVA NORMA", "📂 CARGAR BACKUP"])
    
    with tab1:
        st.caption("Pega aquí el texto. El sistema detectará Títulos y Capítulos automáticamente.")
        axis_input = st.text_input("Eje Temático:", value=engine.thematic_axis)
        txt = st.text_area("Texto de la Norma:", height=150)
        
        if st.button("🚀 PROCESAR Y SEGMENTAR"):
            if engine.process_law(txt, axis_input): 
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.success(f"¡Norma Procesada! Se encontraron {len(engine.sections_map)} secciones.")
                time.sleep(1)
                st.rerun()

    with tab2:
        st.caption("Carga un archivo .json guardado previamente.")
        upl = st.file_uploader("Archivo JSON:", type=['json'])
        if upl is not None:
            if 'last_loaded' not in st.session_state or st.session_state.last_loaded != upl.name:
                try:
                    d = json.load(upl)
                    engine.chunks = d['chunks']
                    engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
                    engine.failed_indices = set(d['failed'])
                    engine.feedback_history = d.get('feed', [])
                    engine.entity = d.get('ent', "")
                    engine.thematic_axis = d.get('axis', "General")
                    engine.level = d.get('lvl', "Profesional")
                    engine.study_phase = d.get('phase', "Pre-Guía")
                    engine.structure_type = d.get('struct_type', "Técnico / Normativo (Sin Caso)")
                    engine.questions_per_case = d.get('q_per_case', 1)
                    engine.example_question = d.get('ex_q', "")
                    engine.job_functions = d.get('job', "")
                    
                    # RECUPERAR SECCIONES
                    engine.sections_map = d.get('sections', {})
                    engine.active_section_name = d.get('act_sec', "Todo el Documento")

                    if DL_AVAILABLE:
                         with st.spinner("🧠 Recuperando memoria neuronal..."):
                            engine.chunk_embeddings = dl_model.encode(engine.chunks)

                    st.session_state.last_loaded = upl.name
                    st.success("¡Backup Cargado!")
                    time.sleep(1)
                    st.session_state.page = 'game'
                    st.session_state.current_data = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al leer: {e}")

    # --- NUEVO SELECTOR DE SECCIÓN (SOLO SI HAY SECCIONES) ---
    if engine.sections_map and len(engine.sections_map) > 1:
        st.divider()
        st.markdown("### 📍 MAPA DE LA LEY")
        # Creamos lista ordenada (keys) pero asegurando que "Todo el Documento" esté primero
        opciones = list(engine.sections_map.keys())
        if "Todo el Documento" in opciones:
            opciones.remove("Todo el Documento")
            opciones.insert(0, "Todo el Documento")
        
        # Recuperar índice actual
        try: idx_sec = opciones.index(engine.active_section_name)
        except: idx_sec = 0
            
        seleccion = st.selectbox("Estudiar Específicamente:", opciones, index=idx_sec)
        
        # Si cambia la selección, actualizamos los chunks
        if seleccion != engine.active_section_name:
            if engine.update_chunks_by_section(seleccion):
                st.session_state.current_data = None # Limpiar pregunta anterior
                st.toast(f"Cambiado a: {seleccion}", icon="✅")
                time.sleep(0.5)
                st.rerun()

    if engine.chunks and engine.api_key and st.session_state.page == 'setup':
        st.divider()
        if st.button("▶️ IR AL SIMULACRO", type="primary"): st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()

    st.divider()
    
    try: lvl_idx = ["Profesional", "Asesor", "Técnico", "Asistencial"].index(engine.level)
    except: lvl_idx = 0
    engine.level = st.selectbox("Nivel:", ["Profesional", "Asesor", "Técnico", "Asistencial"], index=lvl_idx)
    
    try: ent_idx = ENTIDADES_CO.index(engine.entity)
    except: ent_idx = 0
    
    ent_selection = st.selectbox("Entidad:", ENTIDADES_CO, index=ent_idx)
    if "Otra" in ent_selection or "Agregar" in ent_selection:
        engine.entity = st.text_input("Nombre Entidad:", value=engine.entity)
    else:
        engine.entity = ent_selection
            
    if st.button("🔥 INICIAR SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True; st.session_state.current_data = None; st.session_state.page = 'game'; st.rerun()
    
    if engine.chunks:
        full_save_data = {
            "chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices),
            "feed": engine.feedback_history, "ent": engine.entity, "axis": engine.thematic_axis,
            "lvl": engine.level, "phase": engine.study_phase, "ex_q": engine.example_question, "job": engine.job_functions,
            "struct_type": engine.structure_type, "q_per_case": engine.questions_per_case,
            # GUARDAR SECCIONES
            "sections": engine.sections_map, "act_sec": engine.active_section_name
        }
        st.download_button("💾 Guardar Progreso", json.dumps(full_save_data), "backup_titan_full.json")

# --- JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    # Mostrar qué sección se está estudiando
    subtitulo = f"SECCIÓN: {engine.active_section_name}" if engine.active_section_name != "Todo el Documento" else "MODO: GENERAL"
    st.markdown(f"**EJE: {engine.thematic_axis.upper()}** | **{subtitulo}**")
    st.progress(perc/100)

    if not st.session_state.get('current_data'):
        tipo = "CASO NARRATIVO" if "Con Caso" in engine.structure_type else "ENUNCIADO TÉCNICO"
        msg = f"🧠 Generando {engine.questions_per_case} pregunta(s) ({tipo}) sobre {engine.active_section_name}..."
        
        with st.spinner(msg):
            data = engine.generate_case()
            if data and "preguntas" in data:
                st.session_state.current_data = data
                st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()
            else:
                err = data.get('error', 'Desconocido')
                st.error(f"Error: {err}"); st.button("Reintentar", on_click=st.rerun)
                st.stop()

    data = st.session_state.current_data
    narrativa = data.get('narrativa_caso','Error')
    st.markdown(f"<div class='narrative-box'><h4>🏛️ {engine.entity}</h4>{narrativa}</div>", unsafe_allow_html=True)
    
    q_list = data.get('preguntas', [])
    if q_list:
        q = q_list[st.session_state.q_idx]
        st.write(f"### Pregunta {st.session_state.q_idx + 1}")
        
        with st.form(key=f"q_{st.session_state.q_idx}"):
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
        
        st.divider()
        with st.expander("🛠️ CALIBRACIÓN MANUAL", expanded=True):
            reasons_map = {
                "Preguntas no tienen que ver con el Caso": "desconexion",
                "Respuesta Incompleta": "recorte",
                "Spoiler": "spoiler",
                "Respuesta Obvia": "respuesta_obvia",
                "Alucinación": "alucinacion",
                "Opciones Desiguales": "sesgo_longitud",
                "Muy Fácil": "pregunta_facil",
                "Repetitivo": "repetitivo",
                "Incoherente": "incoherente"
            }
            r = st.selectbox("¿Qué estuvo mal?", list(reasons_map.keys()))
            if st.button("¡Castigar y Corregir!"):
                code = reasons_map[r]
                engine.feedback_history.append(code)
                st.toast(f"Calibración enviada: {code}", icon="🛡️")