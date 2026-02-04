import streamlit as st
import google.generativeai as genai
import json
import random
import time
import requests
import re
from collections import Counter

# ==========================================
# GESTIÓN DE DEPENDENCIAS Y MOTORES NEURONALES
# ==========================================
# Intentamos cargar librerías de IA avanzada si están disponibles
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# ==========================================
# CONFIGURACIÓN VISUAL Y ESTILOS CSS
# ==========================================
st.set_page_config(
    page_title="TITÁN v76 - Full Master", 
    page_icon="🦅", 
    layout="wide"
)

st.markdown("""
<style>
    /* Estilo para botones principales en negro elegante */
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        font-weight: bold; 
        height: 3.5em; 
        transition: all 0.3s; 
        background-color: #000000; 
        color: white;
    }
    
    /* Caja para la narrativa del caso/norma */
    .narrative-box {
        background-color: #f5f5f5; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 6px solid #424242; 
        margin-bottom: 25px;
        font-family: 'Georgia', serif; 
        font-size: 1.15em; 
        line-height: 1.6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Etiquetas para artículos fallados (ROJO) */
    .failed-tag {
        background-color: #ffcccc; 
        color: #990000; 
        padding: 4px 8px; 
        border-radius: 4px; 
        font-size: 0.9em; 
        font-weight: bold; 
        margin-right: 5px;
        border: 1px solid #cc0000; 
        display: inline-block;
        margin-bottom: 5px;
    }

    /* Etiquetas para artículos dominados (VERDE) */
    .mastered-tag {
        background-color: #ccffcc; 
        color: #006600; 
        padding: 4px 8px; 
        border-radius: 4px; 
        font-size: 0.9em; 
        font-weight: bold; 
        margin-right: 5px;
        border: 1px solid #006600; 
        display: inline-block;
        margin-bottom: 5px;
    }
    
    /* Cajas estadísticas del tablero */
    .stat-box {
        text-align: center; 
        padding: 10px; 
        background: #ffffff; 
        border-radius: 8px; 
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CARGA DEL MODELO DE EMBEDDINGS (Cacheado)
# ==========================================
@st.cache_resource
def load_embedding_model():
    if DL_AVAILABLE: 
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

dl_model = load_embedding_model()

# ==========================================
# LISTA MAESTRA DE ENTIDADES COLOMBIANAS
# ==========================================
ENTIDADES_CO = [
    "Contraloría General de la República", 
    "Fiscalía General de la Nación",
    "Procuraduría General de la Nación", 
    "Defensoría del Pueblo",
    "DIAN", 
    "Registraduría Nacional", 
    "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia", 
    "Consejo de Estado", 
    "Corte Constitucional",
    "Policía Nacional", 
    "Ejército Nacional", 
    "ICBF", 
    "SENA", 
    "Ministerio de Educación", 
    "Ministerio de Salud", 
    "DANE",
    "Otra (Manual) / Agregar +"
]

# ==========================================
# CLASE PRINCIPAL: MOTOR JURÍDICO TITÁN
# ==========================================
class LegalEngineTITAN:
    def __init__(self):
        # -- Almacenamiento de Datos --
        self.chunks = []           
        self.chunk_embeddings = None 
        self.mastery_tracker = {}  
        self.failed_indices = set()
        self.feedback_history = [] 
        self.current_data = None
        self.current_chunk_idx = -1
        
        # -- Configuración de Usuario --
        self.entity = ""
        self.level = "Profesional" 
        self.simulacro_mode = False
        self.provider = "Unknown" 
        self.api_key = ""
        self.model = None 
        self.current_temperature = 0.3 
        self.last_failed_embedding = None
        
        # -- Variables de Control Pedagógico --
        self.study_phase = "Pre-Guía" 
        self.example_question = "" 
        self.job_functions = ""    
        self.thematic_axis = "General"
        self.structure_type = "Técnico / Normativo (Sin Caso)" 
        self.questions_per_case = 1 
        
        # -- Mapa de la Ley (Jerarquía) --
        self.sections_map = {} 
        self.active_section_name = "Todo el Documento"
        
        # -- Sistema Francotirador & Semáforo --
        self.seen_articles = set()    
        self.failed_articles = set()   # Lista Roja (Pendientes)
        self.mastered_articles = set() # Lista Verde (Dominados)
        self.temporary_blacklist = set() # Lista Negra de Sesión
        self.current_article_label = "General"

    # ------------------------------------------
    # CONFIGURACIÓN DE API (LLAVE MAESTRA)
    # ------------------------------------------
    def configure_api(self, key):
        key = key.strip()
        self.api_key = key
        
        if key.startswith("gsk_"):
            self.provider = "Groq"
            return True, "🚀 Motor GROQ Activado"
        elif key.startswith("sk-") or key.startswith("sk-proj-"): 
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

    # ------------------------------------------
    # SEGMENTACIÓN INTELIGENTE v69 (VISIÓN ÁGUILA)
    # ------------------------------------------
    def smart_segmentation(self, full_text):
        """
        Divide el texto respetando la jerarquía: 
        Libro > Título > Capítulo > Sección > Artículo.
        (Versión v69: Busca nombres en líneas siguientes si el título es corto)
        """
        lineas = full_text.split('\n')
        secciones = {"Todo el Documento": []} 
        
        active_hierarchy = {
            "LIBRO": None, "TÍTULO": None, "CAPÍTULO": None, "SECCIÓN": None, "ARTÍCULO": None
        }

        # --- PATRONES REGEX FLEXIBLES ---
        patron_libro = r'^\s*(LIBRO)\.?\s+[IVXLCDM]+\b'
        patron_titulo_romano = r'^\s*([IVXLCDM]+)\.\s+(.+)' 
        patron_titulo_txt = r'^\s*(TÍTULO|TITULO)\.?\s+[IVXLCDM]+\b' 
        patron_capitulo = r'^\s*(CAPÍTULO|CAPITULO)\.?\s+[IVXLCDM0-9]+\b'
        patron_seccion_txt = r'^\s*(SECCIÓN|SECCION)\.?\s+'
        patron_articulo = r'^\s*(ARTÍCULO|ARTICULO|ART)\.?\s*\d+'

        # --- FUNCIÓN AUXILIAR v69: MIRAR ABAJO ---
        def buscar_continuacion(idx_actual):
            # Mira hasta 3 líneas abajo buscando texto que no sea otro título
            for i in range(1, 4): 
                if idx_actual + i < len(lineas):
                    txt = lineas[idx_actual + i].strip()
                    if txt: # Si hay texto
                        # Si es otro encabezado (Art, Cap, etc), paramos y no unimos
                        if re.match(r'^(ART|CAP|TIT|LIB|SEC)', txt, re.IGNORECASE): 
                            return None
                        return txt # Encontramos el nombre (Ej: "DERECHO DE PETICIÓN")
            return None

        for idx, linea in enumerate(lineas):
            linea_limpia = linea.strip()
            if not linea_limpia: continue

            # Detectar LIBRO
            if re.match(patron_libro, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60: # Si es corto, busca nombre abajo
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                
                active_hierarchy["LIBRO"] = label
                active_hierarchy["TÍTULO"] = None; active_hierarchy["CAPÍTULO"] = None; active_hierarchy["SECCIÓN"] = None
                secciones[label] = []

            # Detectar TÍTULO
            elif re.match(patron_titulo_romano, linea_limpia, re.IGNORECASE) or re.match(patron_titulo_txt, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60: # Si es corto (Ej: TÍTULO II), busca nombre abajo
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                
                active_hierarchy["TÍTULO"] = label
                active_hierarchy["CAPÍTULO"] = None; active_hierarchy["SECCIÓN"] = None
                secciones[label] = []

            # Detectar CAPÍTULO
            elif re.match(patron_capitulo, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60:
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                
                active_hierarchy["CAPÍTULO"] = label; active_hierarchy["SECCIÓN"] = None
                secciones[label] = []

            # Detectar SECCIÓN
            elif re.match(patron_seccion_txt, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60:
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                
                active_hierarchy["SECCIÓN"] = label
                secciones[label] = []
            
            # Detectar ARTÍCULO
            elif re.match(patron_articulo, linea_limpia, re.IGNORECASE):
                label = linea_limpia.split('.')[0] + "."
                if len(label) > 20: label = label[:20]
                active_hierarchy["ARTÍCULO"] = label

            # Guardado en cascada (Nesting)
            secciones["Todo el Documento"].append(linea) 
            if active_hierarchy["LIBRO"]: secciones[active_hierarchy["LIBRO"]].append(linea)
            if active_hierarchy["TÍTULO"]: secciones[active_hierarchy["TÍTULO"]].append(linea)
            if active_hierarchy["CAPÍTULO"]: secciones[active_hierarchy["CAPÍTULO"]].append(linea)
            if active_hierarchy["SECCIÓN"]: secciones[active_hierarchy["SECCIÓN"]].append(linea)

        return {k: "\n".join(v) for k, v in secciones.items() if v}

    # ------------------------------------------
    # PROCESAMIENTO DE TEXTO (CHUNKS)
    # ------------------------------------------
    def process_law(self, text, axis_name):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        self.thematic_axis = axis_name 
        self.sections_map = self.smart_segmentation(text)
        
        # Chunks de 50.000 caracteres para mantener contexto amplio
        self.chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
        self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
        
        if dl_model: 
            with st.spinner("🧠 Analizando jerarquía..."): 
                self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(self.chunks)

    def update_chunks_by_section(self, section_name):
        if section_name in self.sections_map:
            texto_seccion = self.sections_map[section_name]
            self.chunks = [texto_seccion[i:i+50000] for i in range(0, len(texto_seccion), 50000)]
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.active_section_name = section_name
            if dl_model: self.chunk_embeddings = dl_model.encode(self.chunks)
            self.seen_articles.clear()
            self.temporary_blacklist.clear() # Limpiar bloqueo al cambiar sección
            return True
        return False

    # ------------------------------------------
    # ESTADÍSTICAS Y PROGRESO
    # ------------------------------------------
    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
        
        # --- AJUSTE MATEMÁTICO (50 Puntos para llenado lento) ---
        SCORE_THRESHOLD = 50
        
        score = sum([min(v, SCORE_THRESHOLD) for v in self.mastery_tracker.values()])
        perc = int((score / (total * SCORE_THRESHOLD)) * 100) if total > 0 else 0
        
        return min(perc, 100), len(self.failed_indices), total

    def get_strict_rules(self):
        return "1. NO SPOILERS: La pregunta NO debe dar la respuesta. 2. DEPENDENCIA: Obligatorio leer el texto."

    def get_calibration_instructions(self):
        return """
        INSTRUCCIONES DE FORMATO:
        1. NO REPETIR TEXTO: El 'enunciado' NO debe repetir lo que ya dice la 'narrativa_caso'.
        2. NO CHIVATEAR: No digas "Según el Título X". Di "Según la norma".
        """

    # ------------------------------------------
    # GENERADOR DE CASOS (MOTOR PRINCIPAL)
    # ------------------------------------------
    def generate_case(self):
        if not self.api_key: return {"error": "Falta Llave"}
        if not self.chunks: return {"error": "Falta Norma"}
        
        idx = -1
        # Lógica de repaso de errores (Embeddings)
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]
        
        if idx == -1: idx = random.choice(range(len(self.chunks)))
        self.current_chunk_idx = idx
        
        texto_base = self.chunks[idx]
        
        # --- FRANCOTIRADOR CON ANCLAJE (Regex Start of Line) ---
        patron_articulo = r'^\s*(?:ARTÍCULO|ARTICULO|ART)\.?\s*(\d+[A-Z]?)'
        matches = list(re.finditer(patron_articulo, texto_base, re.IGNORECASE | re.MULTILINE))
        
        texto_final_ia = texto_base
        etiqueta_articulo = "General / Sin Artículo Detectado"
        
        if matches:
            # FILTRO v72: Filtrar Vistos Y BLOQUEADOS
            candidatos = [m for m in matches if m.group(0).upper().strip() not in self.seen_articles and m.group(0).upper().strip() not in self.temporary_blacklist]
            
            if not candidatos:
                # Si todos están bloqueados o vistos, reseteamos vistos pero MANTENEMOS los bloqueados si es posible
                candidatos = [m for m in matches if m.group(0).upper().strip() not in self.temporary_blacklist]
                if not candidatos: # Si TODO está bloqueado, liberamos
                    candidatos = matches
                    self.temporary_blacklist.clear()
                self.seen_articles.clear()
            
            # Selección aleatoria
            seleccion = random.choice(candidatos)
            
            # --- FOCUS ESTRICTO (CORTE EXACTO) ---
            start_pos = seleccion.start()
            current_match_index = matches.index(seleccion)
            
            # Si existe un siguiente artículo, cortamos ANTES de que empiece
            if current_match_index + 1 < len(matches):
                end_pos = matches[current_match_index + 1].start()
            else:
                end_pos = min(len(texto_base), start_pos + 4000)

            texto_final_ia = texto_base[start_pos:end_pos] 
            etiqueta_articulo = seleccion.group(0).upper().strip()
            self.current_article_label = etiqueta_articulo

            # --- MICRO-SEGMENTACIÓN v76 (MEJORADA: Numerales + Definiciones) ---
            # Busca:
            # 1. Numerales (1., 2.)
            # 2. Literales (a), b))
            # 3. Definiciones con : o . (Ej: "Vigilancia fiscal.")
            patron_item = r'(^\s*\d+\.\s+|^\s*[a-z]\)\s+|^\s*[A-Z][a-zA-Z\s\u00C0-\u00FF]{2,50}[:\.])'
            
            sub_matches = list(re.finditer(patron_item, texto_final_ia, re.MULTILINE))
            
            if len(sub_matches) > 1:
                # Elegimos uno al azar para preguntar específicamente sobre él
                sel_sub = random.choice(sub_matches)
                start_sub = sel_sub.start()
                idx_sub = sub_matches.index(sel_sub)
                
                # Cortar hasta el siguiente ítem o fin del texto
                end_sub = sub_matches[idx_sub+1].start() if idx_sub + 1 < len(sub_matches) else len(texto_final_ia)
                
                texto_fragmento = texto_final_ia[start_sub:end_sub]
                
                # Limpiamos el ID (quitamos espacios y puntos finales para la etiqueta)
                id_sub = sel_sub.group(0).strip()
                if len(id_sub) > 20: id_sub = id_sub[:20] + "..."
                
                # Contexto: Mantenemos el encabezado del artículo + el fragmento
                encabezado = texto_final_ia[:100].split('\n')[0] 
                if "ARTÍCULO" not in encabezado.upper(): encabezado = f"{self.current_article_label} (Contexto)"
                
                texto_final_ia = f"{encabezado}\n[...]\n{texto_fragmento}"
                self.current_article_label = f"{self.current_article_label} - ITEM {id_sub}"

        else:
            self.current_article_label = "General"
            texto_final_ia = texto_base[:4000]

        # Configuración de Nivel
        dificultad_prompt = ""
        if self.level == "Asistencial":
            dificultad_prompt = "NIVEL: ASISTENCIAL. Preguntas de memoria, archivo y plazos exactos."
        elif self.level == "Técnico":
            dificultad_prompt = "NIVEL: TÉCNICO. Aplicación de procesos y requisitos."
        elif self.level == "Profesional":
            dificultad_prompt = "NIVEL: PROFESIONAL. Alta dificultad. Análisis de casos y principios. Prohibido preguntas obvias."
        elif self.level == "Asesor":
            dificultad_prompt = "NIVEL: ASESOR. Muy Alta dificultad. Estrategia y jurisprudencia."

        instruccion_estilo = "ESTILO: TÉCNICO. 'narrativa_caso' = Contexto normativo." if "Sin Caso" in self.structure_type else "ESTILO: NARRATIVO. Historia laboral realista."

        # --- RECONEXIÓN DE CALIBRACIÓN (v76: 5 CAPITANES) ---
        feedback_instr = ""
        if self.feedback_history:
            last_feeds = self.feedback_history[-5:] # Tomamos los últimos 5 reclamos
            instrucciones_correccion = []
            
            if "pregunta_facil" in last_feeds: 
                instrucciones_correccion.append("ALERTA: El usuario reportó 'Muy Fácil'. AUMENTAR DRASTICAMENTE LA DIFICULTAD Y COMPLEJIDAD.")
            if "respuesta_obvia" in last_feeds: 
                instrucciones_correccion.append("ALERTA: El usuario reportó 'Respuesta Obvia'. USAR OPCIONES TRAMPA OBLIGATORIAS. PROHIBIDO RESPUESTAS EVIDENTES.")
            if "spoiler" in last_feeds: 
                instrucciones_correccion.append("ALERTA: El usuario reportó 'Spoiler'. EL ENUNCIADO NO PUEDE CONTENER PISTAS DE LA RESPUESTA.")
            if "desconexion" in last_feeds: 
                instrucciones_correccion.append("ALERTA: El usuario reportó 'Desconexión'. LA PREGUNTA DEBE ESTAR 100% VINCULADA AL CASO Y TEXTO.")
            if "sesgo_longitud" in last_feeds: # NUEVO CAPITÁN AGREGADO
                instrucciones_correccion.append("ALERTA: El usuario reportó 'Opciones Desiguales'. LA RESPUESTA CORRECTA NO PUEDE SER LA MÁS LARGA. EQUILIBRAR LONGITUD DE TODAS LAS OPCIONES.")
            
            if instrucciones_correccion:
                feedback_instr = "CORRECCIONES DEL USUARIO (PRIORIDAD MAXIMA): " + " ".join(instrucciones_correccion)

        # PROMPT FINAL v74 (CON FEEDBACK REAL + TIP + MICROSEGMENTACION)
        prompt = f"""
        ACTÚA COMO EXPERTO EN CONCURSOS (NIVEL {self.level.upper()}).
        ENTIDAD: {self.entity.upper()}.
        
        {dificultad_prompt}
        {instruccion_estilo}
        {feedback_instr}
        
        Genera {self.questions_per_case} preguntas basándote EXCLUSIVAMENTE en el texto proporcionado abajo.
        
        REGLAS DE OBLIGATORIO CUMPLIMIENTO:
        1. CANTIDAD DE OPCIONES: Genera SIEMPRE 4 opciones de respuesta (A, B, C, D).
        2. ESTILO DEL USUARIO: Si hay un ejemplo abajo, COPIA su estructura de redacción y conectores.
        3. FOCO: No inventes artículos que no estén en el fragmento.
        4. TIP MEMORIA: Incluye un campo 'tip_memoria' con una frase corta, mnemotecnia o palabra clave.
        
        IMPORTANTE - FORMATO DE EXPLICACIÓN (ESTRUCTURADO):
        No me des la explicación en un solo texto corrido.
        Dame un OBJETO JSON llamado "explicaciones" donde cada letra (A, B, C, D) tenga su propia explicación individual.
        Ejemplo: "A": "Es incorrecta porque...", "B": "Es correcta ya que..."
        
        EJEMPLO A IMITAR (ESTILO Y FORMATO):
        '''{self.example_question}'''
        
        NORMA (Fragmento de Estudio): "{texto_final_ia}"
        
        {self.get_strict_rules()}
        {self.get_calibration_instructions()}
        
        FORMATO JSON OBLIGATORIO:
        {{
            "articulo_fuente": "ARTÍCULO X",
            "narrativa_caso": "Texto de contexto...",
            "preguntas": [
                {{
                    "enunciado": "Pregunta...", 
                    "opciones": {{
                        "A": "...", 
                        "B": "...", 
                        "C": "...", 
                        "D": "..."
                    }}, 
                    "respuesta": "A", 
                    "tip_memoria": "Frase mnemotécnica...",
                    "explicaciones": {{
                        "A": "Texto justificando A...",
                        "B": "Texto justificando B...",
                        "C": "Texto justificando C...",
                        "D": "Texto justificando D..."
                    }}
                }}
            ]
        }}
        """
        
        max_retries = 3
        attempts = 0
        while attempts < max_retries:
            try:
                # --- LLAMADA A OPENAI ---
                if self.provider == "OpenAI":
                    headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    data = {
                        "model": "gpt-4o", 
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant. OUTPUT JSON ONLY."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.current_temperature,
                        "response_format": {"type": "json_object"}
                    }
                    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
                    if resp.status_code != 200: return {"error": f"OpenAI Error {resp.status_code}: {resp.text}"}
                    text_resp = resp.json()['choices'][0]['message']['content']

                # --- LLAMADA A GOOGLE ---
                elif self.provider == "Google":
                    safety = [{"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_NONE"} for c in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]]
                    res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "temperature": self.current_temperature}, safety_settings=safety)
                    text_resp = res.text.strip()
                
                # --- LLAMADA A GROQ ---
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
                
                final_json = json.loads(text_resp)
                
                # --- AUTO-FUENTE ---
                if "articulo_fuente" in final_json:
                    self.current_article_label = final_json["articulo_fuente"].upper()
                
                # Si hicimos micro-segmentación, intentamos mantener la etiqueta precisa
                if "ITEM" in self.current_article_label and "ITEM" not in final_json.get("articulo_fuente", "").upper():
                     pass
                elif "articulo_fuente" in final_json:
                     self.current_article_label = final_json["articulo_fuente"].upper()

                # --- BARAJADOR AUTOMÁTICO INTELIGENTE ---
                for q in final_json['preguntas']:
                    opciones_raw = list(q['opciones'].items()) 
                    explicaciones_raw = q.get('explicaciones', {})
                    respuesta_correcta_texto = q['opciones'][q['respuesta']]
                    tip_memoria = q.get('tip_memoria', "")
                    
                    items_barajados = []
                    for k, v in opciones_raw:
                        items_barajados.append({
                            "texto": v,
                            "explicacion": explicaciones_raw.get(k, "Sin detalle."),
                            "es_correcta": (v == respuesta_correcta_texto)
                        })
                    
                    random.shuffle(items_barajados)
                    
                    nuevas_opciones = {}
                    nueva_letra_respuesta = "A"
                    texto_final_explicacion = ""
                    letras = ['A', 'B', 'C', 'D']
                    
                    for i, item in enumerate(items_barajados):
                        if i < 4:
                            letra = letras[i]
                            nuevas_opciones[letra] = item["texto"]
                            
                            estado = "❌ INCORRECTA"
                            if item["es_correcta"]:
                                nueva_letra_respuesta = letra
                                estado = "✅ CORRECTA"
                            
                            texto_final_explicacion += f"**({letra}) {estado}:** {item['explicacion']}\n\n"
                    
                    q['opciones'] = nuevas_opciones
                    q['respuesta'] = nueva_letra_respuesta
                    q['explicacion'] = texto_final_explicacion
                    q['tip_final'] = tip_memoria

                return final_json

            except Exception as e:
                time.sleep(1); attempts += 1
                if attempts == max_retries: return {"error": f"Fallo Crítico: {str(e)}"}
        return {"error": "Saturado."}

# ==========================================
# INTERFAZ DE USUARIO (SIDEBAR Y MAIN)
# ==========================================
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'case_id' not in st.session_state: st.session_state.case_id = 0 # ID Único para evitar fantasmas
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

with st.sidebar:
    st.title("🦅 TITÁN v76 (Master)")
    
    with st.expander("🔑 LLAVE MAESTRA", expanded=True):
        key = st.text_input("API Key (Cualquiera):", type="password")
        if key:
            ok, msg = engine.configure_api(key)
            if ok: st.success(msg)
            else: st.error(msg)
    
    st.divider()
    
    # --- VISUALIZACIÓN DE SEMÁFORO ---
    if engine.failed_articles:
        st.markdown("### 🔴 REPASAR (PENDIENTES)")
        html_fail = ""
        for fail in engine.failed_articles:
            html_fail += f"<span class='failed-tag'>{fail}</span>"
        st.markdown(html_fail, unsafe_allow_html=True)
        
    if engine.mastered_articles:
        st.markdown("### 🟢 DOMINADOS (CONTROL TOTAL)")
        html_master = ""
        for master in engine.mastered_articles:
            html_master += f"<span class='mastered-tag'>{master}</span>"
        st.markdown(html_master, unsafe_allow_html=True)
        
    if engine.failed_articles or engine.mastered_articles:
        st.divider()

    st.markdown("### 📋 ESTRATEGIA")
    fase_default = 0 if engine.study_phase == "Pre-Guía" else 1
    fase = st.radio("Fase:", ["Pre-Guía", "Post-Guía"], index=fase_default)
    engine.study_phase = fase

    st.markdown("#### 🔧 ESTRUCTURA")
    col1, col2 = st.columns(2)
    with col1:
        idx_struct = 0 if "Sin Caso" in engine.structure_type else 1
        estilo = st.radio("Enunciado:", ["Técnico / Normativo (Sin Caso)", "Narrativo / Situacional (Con Caso)"], index=idx_struct)
        engine.structure_type = estilo
    with col2:
        cant = st.number_input("Preguntas:", min_value=1, max_value=5, value=engine.questions_per_case)
        engine.questions_per_case = cant

    with st.expander("Detalles", expanded=True):
        if "Con Caso" in estilo:
            engine.job_functions = st.text_area("Funciones / Rol:", value=engine.job_functions, height=70, placeholder="Ej: Profesional Universitario...")
        else:
            engine.example_question = st.text_area("Ejemplo de Estilo (Sintaxis):", value=engine.example_question, height=70, placeholder="Pega el ejemplo para copiar los 'dos puntos' y conectores...")

    st.divider()
    
    tab1, tab2 = st.tabs(["📝 NUEVA NORMA", "📂 CARGAR BACKUP"])
    
    with tab1:
        st.caption("Pega aquí el texto. El sistema detectará Jerarquía Completa (Flexible).")
        axis_input = st.text_input("Eje Temático (Ej: Ley 1755):", value=engine.thematic_axis)
        txt = st.text_area("Texto de la Norma:", height=150)
        
        if st.button("🚀 PROCESAR Y SEGMENTAR"):
            if engine.process_law(txt, axis_input): 
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.success(f"¡Norma Procesada! {len(engine.sections_map)} secciones maestras.")
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
                    engine.sections_map = d.get('sections', {})
                    engine.active_section_name = d.get('act_sec', "Todo el Documento")
                    
                    # Recuperar datos
                    engine.seen_articles = set(d.get('seen_arts', []))
                    engine.failed_articles = set(d.get('failed_arts', []))
                    engine.mastered_articles = set(d.get('mastered_arts', []))

                    if DL_AVAILABLE:
                         with st.spinner("🧠 Recuperando memoria neuronal..."): engine.chunk_embeddings = dl_model.encode(engine.chunks)

                    st.session_state.last_loaded = upl.name
                    st.success("¡Backup Cargado!")
                    time.sleep(1); st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
                except Exception as e: st.error(f"Error al leer: {e}")

    if engine.sections_map and len(engine.sections_map) > 1:
        st.divider()
        st.markdown("### 📍 MAPA DE LA LEY")
        opciones = list(engine.sections_map.keys())
        if "Todo el Documento" in opciones: opciones.remove("Todo el Documento"); opciones.insert(0, "Todo el Documento")
        
        try: idx_sec = opciones.index(engine.active_section_name)
        except: idx_sec = 0
            
        seleccion = st.selectbox("Estudiar Específicamente:", opciones, index=idx_sec)
        
        if seleccion != engine.active_section_name:
            if engine.update_chunks_by_section(seleccion):
                st.session_state.current_data = None
                st.toast(f"Cambiado a: {seleccion}", icon="✅")
                time.sleep(0.5); st.rerun()

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
    if "Otra" in ent_selection or "Agregar" in ent_selection: engine.entity = st.text_input("Nombre Entidad:", value=engine.entity)
    else: engine.entity = ent_selection
            
    if st.button("🔥 INICIAR SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True; st.session_state.current_data = None; st.session_state.page = 'game'; st.rerun()
    
    if engine.chunks:
        full_save_data = {
            "chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices),
            "feed": engine.feedback_history, "ent": engine.entity, "axis": engine.thematic_axis,
            "lvl": engine.level, "phase": engine.study_phase, "ex_q": engine.example_question, "job": engine.job_functions,
            "struct_type": engine.structure_type, "q_per_case": engine.questions_per_case,
            "sections": engine.sections_map, "act_sec": engine.active_section_name,
            # Guardar datos nuevos
            "seen_arts": list(engine.seen_articles), "failed_arts": list(engine.failed_articles), "mastered_arts": list(engine.mastered_articles)
        }
        st.download_button("💾 Guardar Progreso", json.dumps(full_save_data), "backup_titan_full.json")

# ==========================================
# CICLO PRINCIPAL DEL JUEGO
# ==========================================
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    subtitulo = f"SECCIÓN: {engine.active_section_name}" if engine.active_section_name != "Todo el Documento" else "MODO: GENERAL"
    
    foco_msg = f"🎯 ENFOQUE CONFIRMADO: **{engine.current_article_label}**"
    st.info(foco_msg)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Dominio Global", f"{perc}%")
    c2.metric("❌ Preguntas Falladas", f"{fails}")
    c3.metric("📉 Bloques Vistos", f"{len([x for x in engine.mastery_tracker.values() if x > 0])}/{total}")

    st.markdown(f"**EJE: {engine.thematic_axis.upper()}** | **{subtitulo}**")
    st.progress(perc/100)

    if not st.session_state.get('current_data'):
        msg = f"🧠 Analizando {engine.current_article_label} - NIVEL {engine.level.upper()}..."
        with st.spinner(msg):
            data = engine.generate_case()
            if data and "preguntas" in data:
                st.session_state.case_id += 1 
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
        
        form_key = f"q_{st.session_state.case_id}_{st.session_state.q_idx}"
        
        with st.form(key=form_key):
            opciones_validas = {k: v for k, v in q['opciones'].items() if v}
            sel = st.radio(q['enunciado'], [f"{k}) {v}" for k,v in opciones_validas.items()], index=None)
            
            # --- BOTONES DE ACCIÓN (v72) ---
            col_val, col_skip = st.columns([1, 1])
            with col_val:
                submitted = st.form_submit_button("✅ VALIDAR RESPUESTA")
            with col_skip:
                skipped = st.form_submit_button("⏭️ SALTAR (BLOQUEAR)")
            
            # LÓGICA DE SALTO (BLOQUEO TEMPORAL)
            if skipped:
                engine.temporary_blacklist.add(engine.current_article_label.split(" - ITEM")[0].strip())
                st.toast(f"🚫 {engine.current_article_label} bloqueado por hoy.", icon="🗑️")
                st.session_state.current_data = None; st.rerun()

            # LÓGICA DE VALIDACIÓN
            if submitted:
                if not sel:
                    st.warning("⚠️ Debes seleccionar una opción primero.")
                else:
                    letra_sel = sel.split(")")[0]
                    full_tag = f"[{engine.thematic_axis}] {engine.current_article_label}"
                    
                    if letra_sel == q['respuesta']: 
                        st.success("✅ ¡Correcto!") 
                        engine.mastery_tracker[engine.current_chunk_idx] += 1
                        
                        if engine.current_article_label != "General":
                            if full_tag in engine.failed_articles: engine.failed_articles.remove(full_tag)
                            engine.mastered_articles.add(full_tag)
                    else: 
                        st.error(f"Incorrecto. Era {q['respuesta']}")
                        engine.failed_indices.add(engine.current_chunk_idx)
                        
                        if engine.chunk_embeddings is not None:
                            engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                        
                        if engine.current_article_label != "General":
                            if full_tag in engine.mastered_articles: engine.mastered_articles.remove(full_tag)
                            engine.failed_articles.add(full_tag)
                    
                    st.info(q['explicacion'])
                    
                    if 'tip_final' in q and q['tip_final']:
                        st.warning(f"💡 **TIP DE MAESTRO:** {q['tip_final']}")
                    
                    st.session_state.answered = True

        if st.session_state.answered:
            if st.session_state.q_idx < len(q_list) - 1:
                if st.button("Siguiente"): st.session_state.q_idx += 1; st.session_state.answered = False; st.rerun()
            else:
                if st.button("Nuevo Caso"): st.session_state.current_data = None; st.rerun()
        
        st.divider()
        with st.expander("🛠️ CALIBRACIÓN MANUAL", expanded=True):
            # --- 5 CAPITANES ACTIVOS (v76) ---
            reasons_map = {
                "Muy Fácil": "pregunta_facil",
                "Respuesta Obvia": "respuesta_obvia",
                "Spoiler (Pistas en enunciado)": "spoiler",
                "Desconexión (Nada que ver)": "desconexion",
                "Opciones Desiguales (Longitud)": "sesgo_longitud"
            }
            errores_sel = st.multiselect("Reportar para ajustar la IA:", list(reasons_map.keys()))
            if st.button("¡Castigar y Corregir!"):
                for r in errores_sel:
                    # Guardamos el código del error
                    engine.feedback_history.append(reasons_map[r])
                st.toast(f"Feedback enviado. IA Ajustada: {len(errores_sel)} correcciones.", icon="🛡️")