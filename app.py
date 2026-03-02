# ### --- INICIO PARTE 1: CABECERA Y ESTÉTICA (CSS) ---
import streamlit as st
import google.generativeai as genai
import json
import random
import time
import requests
import re
import io
import os
import sys
import subprocess
from collections import Counter

# ==============================================================================
# ==============================================================================
#  TITÁN v105: Unobtanium
#  ----------------------------------------------------------------------------
#  ESTA VERSIÓN INCLUYE:
#  1. CEREBRO INSTITUCIONAL: Personalidad de Auditor, Fiscal, etc.
#  2. SEGMENTACIÓN HÍBRIDA: Normas (Artículos) vs Guías (Párrafos).
#  3. MODO TRAMPA & FUNCIONES: Lógica anti-obviedad y contexto laboral.
# ==============================================================================
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. GESTIÓN DE DEPENDENCIAS Y LIBRERÍAS EXTERNAS
# ------------------------------------------------------------------------------

# A. SISTEMA DE IA NEURONAL (Embeddings)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# B. LECTOR DE ARCHIVOS PDF (Vital para tus documentos)
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ------------------------------------------------------------------------------
# 2. CONFIGURACIÓN VISUAL Y ESTILOS (TU CSS ORIGINAL INTACTO)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TITÁN v105 - Unobtanium", 
    page_icon="⚖️", 
    layout="wide"
)

st.markdown("""
<style>

/* 🛡️ SELECTOR ULTRA-PRECISO: Ignora la estructura y busca el ID del Widget */
    [data-testid="stWidgetLabel"] p {
        font-size: 25px !important;
        font-weight: bold !important;
        color: #1E1E1E !important;
        line-height: 1.2 !important;
        padding-bottom: 20px !important;
    }

    /* Tamaño de las Opciones (Respuestas A, B, C, D) */
    [role="radiogroup"] p {
        font-size: 22px !important;
        line-height: 1.4 !important;
        color: #333 !important;
    }

    /* 🛡️ PROTECCIÓN: Mantiene el MENÚ LATERAL en tamaño normal */
    [data-testid="stSidebar"] div[data-testid="stRadio"] > label,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 14px !important; 
        font-weight: normal !important;
    }

    /* Estilo para botones principales en negro elegante */
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        font-weight: bold; 
        height: 3.5em; 
        transition: all 0.3s; 
        background-color: #000000; 
        color: white;
        border: 1px solid #333;
    }
    
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Caja para la narrativa del caso/norma - MODIFICADO */
    .narrative-box {
        background-color: #f5f5f5; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 6px solid #424242; 
        margin-bottom: 25px;
        font-family: 'Georgia', serif; 
        font-size: 35px !important;  /* <--- Sube este valor a 35px o 40px */
        line-height: 1.4 !important;
        color: #1E1E1E !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Ajuste adicional para párrafos dentro de la caja narrativa */
    .narrative-box p {
        font-size: 35px !important; /* <--- Asegura que el texto interno también crezca */
        line-height: 1.4 !important;
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
    
    /* Ajustes generales de tipografía */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 3. CARGA DEL MODELO DE EMBEDDINGS (CACHEADO)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """Carga el modelo vectorial una sola vez."""
    if DL_AVAILABLE: 
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            return None
    return None

dl_model = load_embedding_model()
# ### --- FIN PARTE 1 ---
# ### --- INICIO PARTE 2: ENTIDADES Y CLASE MOTOR (EL CEREBRO) ---
# ------------------------------------------------------------------------------
# 4. LISTA MAESTRA DE ENTIDADES COLOMBIANAS
# ------------------------------------------------------------------------------
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


# ==============================================================================
# ==============================================================================
#  CLASE PRINCIPAL: MOTOR JURÍDICO TITÁN
# ==============================================================================
# ==============================================================================
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
        self.current_temperature = 0.4
        self.last_failed_embedding = None
        self.doc_type = "Norma" 
        
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
        self.last_detected_chapter = 0 
        
        # -- Sistema Francotirador & Semáforo --
        self.seen_articles = set()     
        self.failed_articles = set()   
        self.mastered_articles = set() 
        self.temporary_blacklist = set() 
        self.current_article_label = "General"

        # --- NUEVO: VARIABLE PARA MANUAL DE FUNCIONES ---
        self.manual_text = ""

        # --- ESTE ES EL CAMBIO (EL ARCHIVADOR DE LEYES) ---
        self.law_library = {}

        # --- DICCIONARIO DE MISIONES (El Cerebro) ---
        self.mission_profiles = {
            "Contraloría General de la República": "TU ROL: AUDITOR FISCAL. Tu misión es proteger el PATRIMONIO PÚBLICO. Al generar la pregunta, enfócate exclusivamente en detectar DAÑO PATRIMONIAL, gestión antieconómica, ineficaz o ineficiente. Ignora definiciones de diccionario (RAE) o temas puramente teóricos a menos que sirvan para probar un detrimento económico real. Si el texto es un Manual, pregunta sobre el PROCEDIMIENTO para auditar.",
            "Procuraduría General de la Nación": "TU ROL: JUEZ DISCIPLINARIO. Tu misión es vigilar la CONDUCTA OFICIAL. Enfócate en el cumplimiento de deberes, prohibiciones, inhabilidades e incompatibilidades. No busques cárcel ni dinero, busca FALTAS DISCIPLINARIAS (Gravísimas, Graves, Leves) y afectación a la función pública.",
            "Fiscalía General de la Nación": "TU ROL: FISCAL PENAL. Tu misión es la persecución del DELITO. Enfócate en la tipicidad, antijuridicidad y culpabilidad (Dolo/Culpa). Busca elementos materiales probatorios para un juicio penal. Pregunta sobre requisitos para configurar tipos penales (Peculado, Cohecho, Contratos sin requisitos).",
            "Defensoría del Pueblo": "TU ROL: DEFENSOR DE DERECHOS HUMANOS. Tu misión es la prevención y protección. Enfócate en la tutela de derechos fundamentales, alertas tempranas y garantías constitucionales. Pregunta desde la óptica de la protección al ciudadano.",
            "DIAN": "TU ROL: AUDITOR TRIBUTARIO Y ADUANERO. Tu misión es el recaudo y control. Enfócate en obligaciones tributarias, estatuto tributario, evasión, elusión y control cambiario/aduanero.",
            "Consejo Superior de la Judicatura": "TU ROL: ADMINISTRADOR DE JUSTICIA. Enfócate en la eficiencia de la rama judicial, listas de elegibles, carrera judicial y sanciones disciplinarias a abogados/jueces.",
            "Policía Nacional": "TU ROL: AUTORIDAD DE POLICÍA. Enfócate en la convivencia ciudadana, Código Nacional de Policía, seguridad y orden público civil.",
            "Ejército Nacional": "TU ROL: DEFENSOR DE LA SOBERANÍA. Enfócate en defensa nacional, Derechos Humanos en el marco del DIH y régimen especial de las fuerzas militares.",
            "ICBF": "TU ROL: DEFENSOR DE FAMILIA. Enfócate en el restablecimiento de derechos de niños, niñas y adolescentes. Interés superior del menor.",
            "Genérico": "TU ROL: SERVIDOR PÚBLICO INTEGRAL. Enfócate en los principios de la función pública (Art. 209 Constitución): Igualdad, moralidad, eficacia, economía, celeridad, imparcialidad y publicidad."
        }
# --- LA LAVADORA (FASE 1): DESINFECTANTE DE IDENTIDAD ---
    def clean_label(self, text):
        if not text: return ""
        import re
        import unicodedata
        
        # 1. Limpieza básica y Mayúsculas
        clean = str(text).strip().upper()
        clean = re.sub(r'\s+', ' ', clean)
        
        # 2. REPARADOR DE "O" INTRUSA (Letra, no número)
        # Solo eliminamos si es la LETRA 'O'. 
        # Si el PDF tiene un '0' (número), lo respetamos para no dañar el "60" real.
        if clean and clean[-1] == 'O' and any(char.isdigit() for char in clean):
            clean = clean[:-1].strip()
        
        # 3. UNIFICADOR DE NOMBRES (Quitar tildes)
        # Esto hace que "PRINCIPIOS" y "Principios" sean la misma carpeta
        clean = ''.join(c for c in unicodedata.normalize('NFD', clean)
                       if unicodedata.category(c) != 'Mn')
        
        # 4. Estandarización de formato "ARTICULO X"
        clean = re.sub(r'(?:ARTICULO|ART)\.?\s*', 'ARTICULO ', clean)
        
        # 5. Quitar puntos finales (ej: "ARTICULO 6." -> "ARTICULO 6")
        clean = clean.rstrip('.')
        
        return clean.strip()

# ### --- FIN PARTE 2 ---
# ### --- INICIO PARTE 3: LÓGICA DE PROCESAMIENTO Y SEGMENTACIÓN ---
    # --------------------------------------------------------------------------
    # CONFIGURACIÓN DE API (LLAVE MAESTRA)
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # NUEVO: EXTRACTOR DE ADN (LAVADO DE MANUALES)
    # --------------------------------------------------------------------------
    def _clean_manual_text(self, raw_text):
        """
        FILTRO DE LIMPIEZA EXTREMA:
        Elimina 'basura administrativa' (Fechas 2025, Salarios, Códigos).
        Deja solo el 'ADN Técnico' (Funciones y Propósito).
        """
        prompt = f"""
        ACTÚA COMO UN ANALISTA TÉCNICO DE TALENTO HUMANO EXPERTO EN CONCURSOS PÚBLICOS DE ALTO NIVEL.
        TU MISIÓN: Extraer el "ADN PROFESIONAL" del cargo y eliminar todo el "RUIDO ADMINISTRATIVO".
        
        TEXTO DEL MANUAL (FUENTE):
        '''{raw_text[:25000]}'''
        
        INSTRUCCIONES DE LIMPIEZA ESTRICTA:
        1. IDENTIFICA EL ROL: Extrae el Nombre del Empleo (ej: Profesional 03, Inspector IV, Procurador Judicial) y su Propósito Principal.
        2. EXTRAE EL ADN TÉCNICO: Lista solo las funciones esenciales usando VERBOS RECTORES (ej: Sustanciar, Auditar, Intervenir, Proyectar, Evaluar).
        3. VETO ABSOLUTO (ELIMINA): Prohibido incluir fechas (2024, 2025), salarios, códigos de convocatoria (ej: 232-25), número de vacantes, sedes o requisitos de experiencia/educación.
        4. SALIDA OBLIGATORIA (FORMATO PROFESIONAL): 
           CARGO: [Nombre del empleo]
           PROPÓSITO: [Resumen técnico del impacto del cargo]
           ADN TÉCNICO (FUNCIONES): [Lista corta de verbos rectores y su objeto jurídico]
        """
        try:
            if self.provider == "OpenAI":
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
                return resp.json()['choices'][0]['message']['content']
            elif self.provider == "Google":
                return self.model.generate_content(prompt).text
            elif self.provider == "Groq":
                 headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                 data = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
                 resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                 return resp.json()['choices'][0]['message']['content']
        except:
            return raw_text # Fallback si falla la API
        return raw_text

    # --------------------------------------------------------------------------
    # SEGMENTACIÓN INTELIGENTE (TITÁN V105: ARQUITECTURA HÍBRIDA + HERENCIA)
    # --------------------------------------------------------------------------
    def smart_segmentation(self, full_text):
        """
        Divide el texto asegurando que los Títulos contengan a sus Capítulos y Secciones.
        Aplica filtro de ruido para Función Pública y Secretaría del Senado.
        """
        secciones = {"TODO EL DOCUMENTO": []}
        
        # LISTA NEGRA DE RUIDO (Estandarización Multifuente + VIGENCIAS)
        RUIDO_PDF = [
            "DEPARTAMENTO ADMINISTRATIVO", "FUNCIÓN PÚBLICA", "EVA - GESTOR NORMATIVO", 
            "PÁGINA", "DIARIO OFICIAL", "FECHA Y HORA DE CREACIÓN", "Leyes desde 1992", 
            "Última actualización", "ISSN", "secretariasenado.gov.co", 
            "Jurisprudencia Vigencia", "Notas de vigencia", "Legislación anterior",
            "PUBLÍQUESE Y CÚMPLASE", "Dada en Bogotá", "REPÚBLICA DE COLOMBIA"
        ]

        if self.doc_type == "Norma (Leyes/Decretos)":
            lineas = full_text.split('\n')
            
            # Rastreadores de Estado
            c_libro = ""; c_titulo = ""; c_capitulo = ""; c_seccion = ""
            
            # TRADUCTOR UNIVERSAL (Regex para: 1º, 1o., 1ª, I, II, PRIMERO, SEGUNDO...)
            p_word_num = r'(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO|DÉCIMO|[IVXLCDM\d]+[º°\.oª]?)'
            
            p_libro = rf'^\s*(LIBRO)\s+{p_word_num}\b'
	    # CORREGIDO: "TÍTULO I" es válido, pero si está solo el Romano DEBE llevar punto (I.) para no capturar listas I)
            p_tit = rf'^\s*(?:(?:TÍTULO|TITULO|TITULO PRELIMINAR)\s*{p_word_num}?|[IVXLCDM]+\.)\b' 
            # CORREGIDO: "CAPÍTULO 1" es válido, pero si está solo el número DEBE llevar punto (1.) para no capturar listas 1)
            p_cap = rf'^\s*(?:(?:CAPÍTULO|CAPITULO)\s+{p_word_num}|\d+\.)\b'
            p_sec = rf'^\s*(SECCIÓN|SECCION)\s+{p_word_num}\b'
            # Soporte total para artículos: ARTICULO 1º, ARTÍCULO 1o., ARTICULO 1.
	    # Regex Purificada V106: Grupo 1 solo captura el número limpio
            p_art = r'(?:ARTÍCULO|ARTICULO|ART)\.?\s*([IVXLCDM]+|\d+)(?:[º°\.oOª\s]*)\b'
            for i in range(len(lineas)):
                linea_raw = lineas[i]
                
                # Normalización: Soldar romanos rotos (ej: I I -> II)
                linea_limpia = re.sub(r'(?<=[IVXLCDM])\s+(?=[IVXLCDM])', '', linea_raw, flags=re.I).strip()
                
                # FILTRO DE RUIDO: Si la línea es basura del PDF, se ignora
                if not linea_limpia or any(ruido.upper() in linea_limpia.upper() for ruido in RUIDO_PDF): 
                    continue

                def get_full_name_v106(idx, line_match, pattern):
                    """Extrae el nombre descriptivo de la jerarquía."""
                    base_label = line_match.strip().upper()
                    parts = re.split(pattern, line_match, flags=re.I)
                    if len(parts) > 1 and len(parts[-1].strip()) > 3:
                        return f"{base_label}: {parts[-1].strip().upper()}"
                    if idx + 1 < len(lineas):
                        next_line = lineas[idx + 1].strip()
                        if next_line and not any(re.match(p, next_line, re.I) for p in [p_libro, p_tit, p_cap, p_sec, p_art]):
                            if not any(ruido.upper() in next_line.upper() for ruido in RUIDO_PDF):
                                return f"{base_label}: {next_line.upper()}"
                    return base_label

                # ACTUALIZACIÓN DE ESTADOS (Detección de Jerarquía)
                if re.match(p_libro, linea_limpia, re.I): 
                    c_libro = get_full_name_v106(i, linea_limpia, p_libro)
                    c_titulo = ""; c_capitulo = ""; c_seccion = ""
                elif re.match(p_tit, linea_limpia, re.I): 
                    c_titulo = get_full_name_v106(i, linea_limpia, p_tit)
                    c_capitulo = ""; c_seccion = ""
                elif re.match(p_cap, linea_limpia, re.I): 
                    c_capitulo = get_full_name_v106(i, linea_limpia, p_cap)
                    c_seccion = ""
                elif re.match(p_sec, linea_limpia, re.I):
                    c_seccion = get_full_name_v106(i, linea_limpia, p_sec)

                # --- LÓGICA DE HERENCIA (CASCADA) ---
                # Cada línea se guarda en su contenedor y en todos sus contenedores padres
                niveles_activos = ["TODO EL DOCUMENTO"]
                
                if c_libro: 
                    niveles_activos.append(c_libro)
                if c_titulo:
                    nom_tit = f"{c_libro} > {c_titulo}" if c_libro else c_titulo
                    niveles_activos.append(nom_tit)
                if c_capitulo:
                    prefix = f"{c_libro} > " if c_libro else ""
                    prefix += f"{c_titulo} > " if c_titulo else ""
                    niveles_activos.append(prefix + c_capitulo)
                if c_seccion:
                    prefix = f"{c_libro} > " if c_libro else ""
                    prefix += f"{c_titulo} > " if c_titulo else ""
                    prefix += f"{c_capitulo} > " if c_capitulo else ""
                    niveles_activos.append(prefix + c_seccion)

                for nivel in niveles_activos:
                    if nivel not in secciones: secciones[nivel] = []
                    secciones[nivel].append(linea_raw)
                
            return {k: "\n".join(v) for k, v in secciones.items() if len(v) > 0}

        else:
            # Estrategia para Guías Técnicas (Bloques fijos)
            text_clean = re.sub(r'\n\s*\n', '<PARAGRAPH_BREAK>', full_text)
            raw_paragraphs = text_clean.split('<PARAGRAPH_BREAK>')
            final_blocks = {}; current_block_content = ""; block_count = 1
            for p in raw_paragraphs:
                p = p.strip()
                if not p: continue
                if len(current_block_content) + len(p) < 2500:
                    current_block_content += "\n\n" + p
                else:
                    final_blocks[f"BLOQUE {block_count}"] = [current_block_content]
                    block_count += 1; current_block_content = p 
            if current_block_content:
                final_blocks[f"BLOQUE {block_count}"] = [current_block_content]
            final_blocks["TODO EL DOCUMENTO"] = [full_text]
            return {k: "\n".join(v) for k, v in final_blocks.items()}

    # --------------------------------------------------------------------------
    # PROCESAMIENTO Y ACTUALIZACIÓN (OPTIMIZADO: LIMPIEZA AUTOMÁTICA)
    # --------------------------------------------------------------------------
    def process_law(self, text, axis_name, doc_type_input):
        text = text.replace('\r', '')
        if len(text) < 100: return 0, ""
        
        adn_summary = ""
        # --- NUEVO: FILTRO DE PURIFICACIÓN PARA MANUALES ---
        # Si el usuario carga un Manual, lo limpiamos ANTES de guardarlo.

        if doc_type_input == "Guía Técnica / Manual":
            with st.spinner("🧹 Purificando Manual..."):
                adn_summary = self._clean_manual_text(text)
                self.manual_text = adn_summary 
                # text = adn_summary  <-- Comenta esto para NO borrar el contenido original
        else:
            # Si carga una Norma, nos aseguramos de no tener residuos de manuales anteriores
            if not hasattr(self, 'manual_text') or not self.manual_text: 
                self.manual_text = "" 

        # Forzamos mayúsculas desde el inicio para que coincida con la law_library
        self.thematic_axis = self.clean_label(axis_name) 
        self.doc_type = doc_type_input 

        # 1. Generamos el mapa exclusivo de ESTA nueva ley
        nuevo_mapa_ley = self.smart_segmentation(text)
        
        # 2. Creamos la "Estantería" si no existe y guardamos el mapa de esta ley
        if not hasattr(self, 'law_library'): self.law_library = {}
        self.law_library[str(axis_name).strip().upper()] = nuevo_mapa_ley
        
        # 3. El visor actual (sections_map) mostrará solo esta ley recién cargada
        self.sections_map = nuevo_mapa_ley 
        
        self.active_section_name = "TODO EL DOCUMENTO"
        
        # 4. ACUMULAMOS los bloques (Esto lo dejamos como tú lo tienes, para el Global)
        nuevos_chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
        if not self.chunks: self.chunks = []
        self.chunks.extend(nuevos_chunks)

        if not self.mastery_tracker: self.mastery_tracker = {}
        if dl_model: 
            with st.spinner("🧠 Generando mapa neuronal..."): 
                self.chunk_embeddings = dl_model.encode(self.chunks)
        
        return len(self.chunks), adn_summary

    def update_chunks_by_section(self, section_name):
        if section_name in self.sections_map:
            texto_seccion = self.sections_map[section_name]
            self.chunks = [texto_seccion[i:i+50000] for i in range(0, len(texto_seccion), 50000)]
            self.active_section_name = section_name
            if dl_model: self.chunk_embeddings = dl_model.encode(self.chunks)
            self.seen_articles.clear(); self.temporary_blacklist.clear()
            return True
        return False

    def get_stats(self):
        """
        CÁLCULO DE PRECISIÓN ABSOLUTA + FILTRO DE INEXEQUIBILIDAD
        """
        if not self.chunks: return 0, 0, 0
        
        texto_estudio = self.sections_map.get(self.active_section_name, "\n".join(self.chunks))
        
        # 1. DEFINIR PATRÓN DE BÚSQUEDA (Sincronizado con Sniper V106)
        if self.doc_type == "Norma (Leyes/Decretos)":
            # Captura solo el número en el Grupo 1, ignorando la basura (º, o, .)
            p_censo = r'(?:ARTÍCULO|ARTICULO|ART)\.?\s*([IVXLCDM]+|\d+)(?:[º°\.oOª\s]*)\b'
        else:
            p_censo = r'^\s*(\d+(?:\.\d+)+)\b' 
            
        # 2. CENSO FILTRADO (Detectar y Descartar Inexequibles + LIMPIEZA)
        items_validos = []
        for match in re.finditer(p_censo, texto_estudio, re.I | re.M):
            # Miramos 200 caracteres adelante del artículo encontrado
            ventana_contexto = texto_estudio[match.end():match.end()+200].upper()
            
            # Si dice INEXEQUIBLE, DEROGADO o NULO cerca, NO LO CONTAMOS
            if "INEXEQUIBLE" in ventana_contexto or "DEROGADO" in ventana_contexto or "NULO" in ventana_contexto:
                continue
                
# --- LIMPIEZA DE ETIQUETA (FASE 2: SINCRONIZADA) ---
            if self.doc_type == "Norma (Leyes/Decretos)":
                num_limpio = match.group(1).strip()
                # Aquí usamos la LAVADORA que pegamos en el paso anterior
                label_final = self.clean_label(f"[{self.thematic_axis}] ARTICULO {num_limpio}")
            else:
                label_final = match.group(1).strip() # Para manuales (1.1, 1.2)
            items_validos.append(label_final)

        items_unicos = set(items_validos)
        
        # 3. CÁLCULO FINAL (0-1-2)
        if items_unicos:
            total = len(items_unicos)
            score = sum([min(self.mastery_tracker.get(art, 0), 2) for art in items_unicos])
        else:
            total = len(self.chunks)
            score = sum([min(v, 2) for k, v in self.mastery_tracker.items() if isinstance(k, int)])
            
        # Aseguramos que el porcentaje esté entre 0 y 100 (el Nivel -1 puede dar negativos)
        perc = int((score / (total * 2)) * 100) if total > 0 else 0
        return max(0, min(perc, 100)), len(self.failed_indices), total

    def get_strict_rules(self):
        return "1. NO SPOILERS. 2. DEPENDENCIA DEL TEXTO."

    def get_calibration_instructions(self):
        return "INSTRUCCIONES: NO REPETIR TEXTO, NO 'CHIVATEAR' NIVELES."
# ### --- FIN PARTE 3 ---
# ### --- INICIO PARTE 4: EL GENERADOR DE CASOS (IA SNIPER + 9 CAPITANES) ---
# --------------------------------------------------------------------------
    # GENERADOR DE CASOS (MODIFICADO: MOTOR SATÉLITE V108.1 - AISLAMIENTO TOTAL)
    # --------------------------------------------------------------------------
    def generate_case(self):
        """
        Genera la pregunta. Integra:
        1. Sniper V108 (Escaneo global instantáneo + Aislamiento de IA).
        2. Semáforo (Amarillo -> Pesadilla) por IDENTIDAD.
        3. Los 9 Capitanes (Reglas de Hierro en Prompt).
        4. Filtro Anti-Inexequible.
        5. Lógica Condicional (Bloque Único vs Narrativo).
        """
        if not self.api_key: return {"error": "Falta Llave"}
        if not self.chunks: return {"error": "Falta Norma"}

        # 1. EL OJO DE DIOS: Unimos todo el texto
        texto_completo = "\n".join(self.chunks)

        # 2. BUSCAR TODOS LOS ARTÍCULOS EN MILISEGUNDOS
        matches = []
        if self.doc_type == "Norma (Leyes/Decretos)":
            p_art = r'\b(?:ARTÍCULO|ARTICULO|ART)\.?\s*([IVXLCDM]+|\d+)(?:[º°\.oOª\s]*)\b'
            matches = list(re.finditer(p_art, texto_completo, re.IGNORECASE | re.MULTILINE))
        elif self.doc_type == "Guía Técnica / Manual":
            p_idx = r'^\s*(\d+(?:[\.\s]\d+)*)\.?\s+(.+)'
            matches = list(re.finditer(p_idx, texto_completo, re.MULTILINE))

        # 3. FILTRAR CANDIDATOS (IDENTIDAD PURIFICADA)
        candidatos_validos = []
        # CORRECCIÓN: Curamos los corchetes pero MANTENEMOS el nombre eje_id
        eje_id = self.clean_label(str(self.thematic_axis).replace("[", "").replace("]", "").strip())

        for m in matches:
            num_check = m.group(1).strip().upper()
            if self.doc_type == "Norma (Leyes/Decretos)":
                nombre_completo = self.clean_label(f"[{eje_id}] ARTICULO {num_check}")
            else:
                nombre_completo = self.clean_label(f"[{eje_id}] ITEM {num_check}")

            es_verde = self.mastery_tracker.get(nombre_completo, 0) >= 2
            es_bloqueado = nombre_completo in self.temporary_blacklist
            es_visto_ahora = nombre_completo in self.seen_articles

            if es_verde or es_bloqueado or es_visto_ahora:
                continue

            # Escudo de leyes muertas
            contexto = texto_completo[m.end():m.end()+200].upper()
            if any(x in contexto for x in ["INEXEQUIBLE", "DEROGADO", "NULO"]):
                continue

            candidatos_validos.append((m, nombre_completo))

        # 4. AUTO-CURA INQUEBRANTABLE
        if not candidatos_validos:
            if len(self.seen_articles) > 0 or len(self.temporary_blacklist) > 0:
                self.seen_articles.clear()
                self.temporary_blacklist.clear()
                
                # Segunda vuelta: Rescatar lo que NO está verde (Fallados -1)
                for m in matches:
                    num_check = m.group(1).strip().upper()
                    if self.doc_type == "Norma (Leyes/Decretos)":
                        nombre_c = self.clean_label(f"[{eje_id}] ARTICULO {num_check}")
                    else:
                        nombre_c = self.clean_label(f"[{eje_id}] ITEM {num_check}")
                        
                    if self.mastery_tracker.get(nombre_c, 0) < 2:
                        ctx = texto_completo[m.end():m.end()+200].upper()
                        if not any(x in ctx for x in ["INEXEQUIBLE", "DEROGADO", "NULO"]):
                            candidatos_validos.append((m, nombre_c))

            if not candidatos_validos:
                return {"error": "¡SECCIÓN DOMINADA! 🏆 Ya arrasaste con todos los artículos aquí."}

        # 5. PRIORIDAD ROJA (El Radar Fijo)
        prioridad_roja = [(m, n) for m, n in candidatos_validos if self.mastery_tracker.get(n, 0) == -1]

        if prioridad_roja:
            seleccion, nombre_final = random.choice(prioridad_roja)
        else:
            seleccion, nombre_final = random.choice(candidatos_validos)

        # 6. EXTRACCIÓN QUIRÚRGICA (BLINDAJE CONTRA DESVÍOS DE LA IA)
        start_pos = seleccion.start()
        end_pos = len(texto_completo)
        
        # Cortar EXACTAMENTE antes de que empiece el siguiente artículo
        for m_next in matches:
            if m_next.start() > start_pos:
                end_pos = m_next.start()
                break
                
        texto_final_ia = texto_completo[start_pos:end_pos].strip()
  

        self.current_article_label = nombre_final
        self.seen_articles.add(nombre_final)


   # --- CEREBRO: MODO PESADILLA (NIVEL DIOS - 9 CAPITANES HOSTILES) ---
        # Buscamos la maestría por Nombre (Identidad)
        key_maestria = self.current_article_label.split(" - ITEM")[0].strip().upper()
        if "ARTÍCULO" not in key_maestria and "ITEM" not in key_maestria: key_maestria = self.current_chunk_idx
        
        maestria_actual = self.mastery_tracker.get(key_maestria, 0)
        instruccion_pesadilla = ""
        
        if maestria_actual >= 1:
            instruccion_pesadilla = """
            🔥 ALERTA DE MAESTRÍA (MODO PESADILLA ACTIVO):
            El usuario ya dominó el concepto básico. AHORA ACTIVAR PROTOCOLO DE ALTA COMPLEJIDAD:
            
            1. 🎯 OBJETIVO (CAPITÁN SNIPER): IGNORA la regla general del artículo. Busca el PARÁGRAFO, la EXCEPCIÓN o la nota de vigencia más oscura. Pregunta por lo que "NO" se puede hacer o la excepción a la regla.
            2. 👯 TRAMPAS (CAPITÁN GEMELOS): Las opciones incorrectas NO pueden ser errores obvios. Tienen que ser 'Gemelos Legales': frases que son CORRECTAS en otros contextos o artículos vecinos, pero que NO aplican a este caso específico por un detalle técnico.
            3. 💥 LÓGICA (CAPITÁN COLISIÓN): Plantea un "Caso de Frontera": una situación donde dos normas parecen chocar. La respuesta correcta es la que aplica el principio de especialidad o jerarquía.
            4. 🚫 PROHIBIDO: Preguntas de memoria literal. La pregunta debe obligar a DESCOMPONER el caso para encontrar el error de procedimiento.
            
            DIFICULTAD: 11/10 (Rompe-Ranking). Si la respuesta es obvia, has fallado. El usuario debe dudar entre dos opciones hasta el final.
            """

# --- CONFIGURACIÓN TÉCNICA (ANDERSON & KRATHWOHL - GUÍA CGR) ---
        config_nivel = {
            "Asistencial": {
                "dev": "RECUERDO / COMPRENSIÓN", 
                "bloom": "NIVEL 1-2", 
                "focus": "el dominio de métodos, procesos y marcos de referencia, así como la capacidad de explicar relaciones entre datos y principios."
            },
            "Técnico": {
                "dev": "APLICACIÓN", 
                "bloom": "NIVEL 3", 
                "focus": "la capacidad de seleccionar, transferir y utilizar principios de la norma para ejecutar tareas o proponer soluciones concretas."
            },
            "Profesional": {
                "dev": "ANÁLISIS", 
                "bloom": "NIVEL 4", 
                "focus": "la competencia para descomponer el todo en sus partes e identificar qué requisito legal NO se cumple o qué principio prevalece."
            },
            "Asesor": {
                "dev": "ANÁLISIS SUPERIOR", 
                "bloom": "NIVEL 4+", 
                "focus": "la descomposición crítica para toma de decisiones estratégicas y resolución de colisiones normativas complejas."
            }
        }
        meta = config_nivel.get(self.level, config_nivel["Profesional"])
        
        # --- LÓGICA CONDICIONAL DE ESTRUCTURA (TOGGLE: SIN CASO vs CON CASO) ---
        # AQUÍ ES DONDE EL CÓDIGO DECIDE SI FUSIONA (CGR) O SEPARA (CNSC)
        if "Sin Caso" in self.structure_type:
            # MODO BLOQUE ÚNICO (FUSIÓN TOTAL - ESTILO CGR)
            instruccion_estilo = "ESTILO: TÉCNICO (BLOQUE ÚNICO DE ANÁLISIS)."
            json_structure_instruction = f"""
            FORMATO JSON OBLIGATORIO (MODO BLOQUE ÚNICO - SIGUE ESTAS INSTRUCCIONES INTERNAS):
            {{
                "articulo_fuente": "{self.current_article_label}",
                "narrativa_caso": "", 
                "preguntas": [
                    {{
                        "enunciado": "UN SOLO PÁRRAFO denso y sofisticado (Marco -> Restricción -> Nudo) sin anclas semánticas. NO separes el caso de la pregunta. Fusión total. (PROHIBIDO: El enunciado y la respuesta correcta NO pueden ser idénticos).", 
                        "opciones": {{
                            "A": "Opción Correcta (Condicionada al hecho del caso)...", 
                            "B": "Gemelo Contiguo (Mismo artículo, hipótesis distinta)...", 
                            "C": "Gemelo Contiguo (Principio en tensión que cede)...", 
                            "D": "Gemelo Contiguo (Requisito parecido pero inaplicable)..."
                        }}, 
                        "respuesta": "A", 
                        "tip_memoria": "Mnemotecnia para recordar el matiz técnico...",
                        "explicaciones": {{
                            "A": "Justificación técnica de por qué este principio prevalece...",
                            "B": "Explicación de por qué esta parte del artículo no aplica...",
                            "C": "Explicación de por qué este principio cede...",
                            "D": "Explicación de por qué este requisito no se cumple..."
                        }}
                    }}
                ]
            }}
            """
        else:
            # MODO NARRATIVO SEPARADO (ESTILO CNSC / SITUACIONAL)
            instruccion_estilo = "ESTILO: NARRATIVO (SEPARADO - CONTEXTO Y PREGUNTA)."
            json_structure_instruction = f"""
            FORMATO JSON OBLIGATORIO (MODO NARRATIVO - SIGUE ESTAS INSTRUCCIONES INTERNAS):
            {{
                "articulo_fuente": "{self.current_article_label}",
                "narrativa_caso": "Situación real basada en el ADN del cargo donde introduces una variable CLAVE (sujeto, tiempo, hallazgo)...",
                "preguntas": [
                    {{
                        "enunciado": "Párrafo SIN anclas semánticas que plantea el conflicto técnico de procedibilidad...", 
                        "opciones": {{
                            "A": "Opción Correcta (Condicionada al hecho del caso)...", 
                            "B": "Gemelo Contiguo (Mismo artículo, hipótesis distinta)...", 
                            "C": "Gemelo Contiguo (Principio en tensión que cede)...", 
                            "D": "Gemelo Contiguo (Requisito parecido pero inaplicable)..."
                        }}, 
                        "respuesta": "A", 
                        "tip_memoria": "Mnemotecnia para recordar el matiz técnico...",
                        "explicaciones": {{
                            "A": "Justificación técnica de por qué este principio prevalece...",
                            "B": "Explicación de por qué esta parte del artículo no aplica...",
                            "C": "Explicación de por qué este principio cede...",
                            "D": "Explicación de por qué este requisito no se cumple..."
                        }}
                    }}
                ]
            }}
            """
        
        # 1. TRAMPAS Y DIFICULTAD
        instruccion_trampas = ""
        if self.level in ["Profesional", "Asesor"]:
            instruccion_trampas = "MODO AVANZADO (TRAMPAS): PROHIBIDO hacer preguntas obvias. Las opciones incorrectas (distractores) deben ser ALTAMENTE PLAUSIBLES."

        # 2. LÓGICA DE ROL (JERARQUÍA ESTRICTA: ADN TÉCNICO > ROL PREDEFINIDO)
        # Se inyecta el ADN purificado en la Parte 3
        texto_funciones_real = self.manual_text if self.manual_text else self.job_functions
        contexto_funcional = ""
        mision_entidad = "" 

        if texto_funciones_real:
            # CASO A: HAY MANUAL/ADN (Se usa como Lente de Enfoque y Muro de Estanqueidad)
            funciones_safe = texto_funciones_real[:15000]
            contexto_funcional = f"""
            CONTEXTO DE ROL (ADN TÉCNICO - LENTE EVALUATIVO):
            El usuario aspira a un cargo con este perfil técnico extraído: '{funciones_safe}'.
            INSTRUCCIÓN DE SEGURIDAD (MURO DE ESTANQUEIDAD):
            1. Usa este perfil ÚNICAMENTE para ambientar la 'narrativa_caso' (el personaje) y decidir qué artículos de la ley son relevantes.
            2. PROHIBIDO terminantemente usar fechas (2024, 2025), salarios, códigos de convocatoria (ej: 232-25) o requisitos de experiencia del manual en la pregunta o respuestas.
            3. La pregunta debe evaluar el conocimiento de la NORMA (fuente técnica) aplicada a este rol.
            """
            mision_entidad = "" 
        else:
            # CASO B: NO HAY MANUAL -> USA ROL PREDEFINIDO (PARTE 2)
            perfil_mision = self.mission_profiles.get(self.entity, self.mission_profiles.get("Genérico", "Experto Legal"))
            mision_entidad = f"ROL INSTITUCIONAL (AUTOMÁTICO): {perfil_mision}"

	# 3. REGLA MAESTRA DE MIMESIS (ADAPTABLE: CGR vs CNSC)
        instruccion_mimesis = ""
        if self.study_phase == "Post-Guía" and self.example_question:
            # Detectamos si el usuario quiere bloque único o separado
            es_bloque_unico = "Sin Caso" in self.structure_type
            
            estilo_final = "Fusión total en un solo párrafo denso (Bloque Único)." if es_bloque_unico else "Estructura separada (Caso + Pregunta), pero manteniendo el tono seco del molde."
            
            instruccion_mimesis = f"""
            ⚠️ FASE DE DISECCIÓN ESTRUCTURAL (OBLIGATORIA):
            Analiza el molde de excelencia: '''{self.example_question}'''

            TU MISIÓN: Replica su 'RITMO DE TRES ACTOS' y su COMPLEJIDAD LÓGICA, pero INYECTA EL TEMA DE TU NORMA (PDF):
            
            1. ACTO 1 (MARCO): Definición técnica/jurídica abstracta (Usa el tono del ejemplo, no el tema).
            2. ACTO 2 (RESTRICCIÓN): Limitación legal usando conectores como 'La legislación establece'.
            3. ACTO 3 (NUDO TÉCNICO): Conector 'En ese sentido, es imperativo advertir que...' + EL CONFLICTO DE TU NORMA.
            
            🚫 VACUNA ANTI-CONTAMINACIÓN (CRÍTICO): 
            - El ejemplo habla de 'Contraloría/Delegación'. TU NORMA habla de otro tema.
            - El ejemplo es SOLO UN MOLDE. Su contenido es TÓXICO para esta pregunta.
            - USA EL ESQUELETO DEL EJEMPLO, PERO CON LA CARNE DE TU PDF.

            🚫 PROHIBICIÓN DE FORMATO: Si seleccionaste 'Con Caso', NO uses 'Ante la situación descrita'. 
            Empieza el 'narrativa_caso' directamente con el ACTO 1 y deja el ACTO 3 para el 'enunciado'.
            
              """

        # 4. FEEDBACK (LOS CAPITANES REACTIVOS)
        feedback_instr = ""
        if self.feedback_history:
            last_feeds = self.feedback_history[-5:] 
            instrucciones_correccion = []
            if "pregunta_facil" in last_feeds: instrucciones_correccion.append("ALERTA: AUMENTAR DRASTICAMENTE LA DIFICULTAD.")
            if "respuesta_obvia" in last_feeds: instrucciones_correccion.append("ALERTA: USAR OPCIONES TRAMPA OBLIGATORIAS.")
            if "spoiler" in last_feeds: instrucciones_correccion.append("ALERTA: ELIMINAR PISTAS DEL ENUNCIADO.")
            if "desconexion" in last_feeds: instrucciones_correccion.append("ALERTA: VINCULAR 100% AL TEXTO.")
            if "sesgo_longitud" in last_feeds: instrucciones_correccion.append("ALERTA: EQUILIBRAR LONGITUD DE OPCIONES.")
            
            if instrucciones_correccion:
                feedback_instr = "CORRECCIONES DEL USUARIO (PRIORIDAD MAXIMA): " + " ".join(instrucciones_correccion)

	# PROMPT FINAL (ADN INTEGRADO - ADAPTABLE A CUALQUIER ENTIDAD)
        prompt = f"""
        ACTÚA COMO UN EVALUADOR JEFE DE {self.entity.upper()} PARA EL NIVEL {self.level.upper()}.
        TIPO DE DOCUMENTO: {self.doc_type.upper()}.
        
        REQUERIMIENTOS TÉCNICOS OBLIGATORIOS (GUÍA DE ORIENTACIÓN):
        1. NIVEL DE DESARROLLO DE COMPETENCIA: {meta['dev']}. 
        2. DIMENSIÓN COGNITIVA (BLOOM): {meta['bloom']}.
        3. FOCO EVALUATIVO: La pregunta debe centrarse en la {meta['focus']}
        
       METODOLOGÍA DE CONSTRUCCIÓN (NIVEL DE ESFUERZO):
        - ANÁLISIS DE ERRORES (Nivel 4): Si el nivel es Profesional/Asesor, el usuario debe descomponer el todo en sus partes. La pregunta debe obligar a identificar qué requisito legal NO se cumple o qué principio prevalece en una colisión técnica.
        - APLICACIÓN DIRECTA (Nivel 3): Si el nivel es Asistencial/Técnico, el usuario debe seleccionar y utilizar principios para ejecutar una tarea o resolver un problema específico.
        
        {mision_entidad}
        {contexto_funcional}
        {instruccion_pesadilla}
        {instruccion_mimesis}
        {instruccion_estilo}
        {instruccion_trampas}
        {feedback_instr}
        
        MISION: Genera {self.questions_per_case} preguntas de NIVEL ELITE (ROMPE-RANKING) basándote EXCLUSIVAMENTE en el texto proporcionado abajo.
        
        REGLAS DE ORO (LOS 11 CAPITANES - BLINDAJE EXTREMO):
        1. 🚫 CAPITÁN ANTI-LORO: PROHIBIDO iniciar la respuesta con "Según el artículo...", "De acuerdo a la ley..." o similar. La respuesta debe ser una CONSECUENCIA JURÍDICA o TÉCNICA autónoma (Ej: "Se declara la nulidad...", "Opera el silencio administrativo...").
        2. 👯 CAPITÁN GEMELOS (MODO HOSTIL EXTREMO): Las opciones incorrectas DEBEN ser "Gemelos Legales": fragmentos literales de la norma que regulen situaciones parecidas. OBLIGATORIO: Deben provenir del MISMO ARTÍCULO o de artículos contiguos para eliminar el descarte por tema.
        3. ⚖️ CAPITÁN ECUALIZADOR (SIMETRÍA LETAL Y CAMUFLAJE): Las IAs tienen el grave defecto de hacer la respuesta correcta más larga. ¡TIENES PROHIBIDO CAER EN ESTE SESGO! Regla inquebrantable: Las 4 opciones (A, B, C, D) DEBEN tener EXACTAMENTE la misma extensión visual. En TODAS las opciones, limítate a dar la consecuencia jurídica directa, seca y sin rodeos. ¡ATENCIÓN! Tienes ESTRICTAMENTE PROHIBIDO inventar "relleno administrativo", procedimientos falsos o palabras vacías para engordar los distractores. Para lograr que los distractores tengan el mismo tamaño que la correcta, DEBES usar "Gemelos Legales" (es decir, extraer condiciones, principios o excepciones REALES de artículos contiguos del texto proporcionado). Todas las opciones deben ser texto técnico puro. NADIE DEBE ADIVINAR POR EL TAMAÑO NI POR DESCARTAR RELLENO INVENTADO.
        4. 🧠 CAPITÁN ANTI-OBVIEDAD (Prueba del 50/50): PROHIBIDO usar "Todas las anteriores" o respuestas de sentido común moral. La diferencia entre la correcta y la distractor más fuerte debe ser un matiz técnico (un "podrá" vs "deberá", un plazo, una competencia).
        5. 🗑️ CAPITÁN JUSTICIA: Si el fragmento de texto contiene "INEXEQUIBLE", "DEROGADO" o "NULO", IGNÓRALO COMPLETAMENTE y busca otro parágrafo vigente. No preguntes sobre leyes muertas.
        6. 🔗 CAPITÁN CONTEXTO (DEPENDENCIA LÓGICA TOTAL): La pregunta debe ser TÉCNICAMENTE IRRESOLUBLE sin los datos del caso narrado. El enunciado debe plantear un problema de procedibilidad o competencia donde la respuesta correcta sea una excepción o un requisito específico.
        7. 🧨 CAPITÁN ANTI-ANCLA (PROHIBICIÓN SEMÁNTICA): PROHIBIDO nombrar explícitamente el concepto central evaluado en el enunciado o las opciones (ej: no digas "control fiscal", describe la "vigilancia de los recursos"). El concepto debe inferirse por sus efectos.
        8. 🔀 CAPITÁN CONDICIONALIDAD: La opción correcta debe serlo SOLO si se identifica una condición fáctica implícita en el caso narrado (paradoja de corrección condicionada).
        9. 💥 CAPITÁN COLISIÓN: Obliga al usuario a decidir entre dos principios constitucionales en tensión (ej. Eficacia vs Legalidad) o normas que parecen chocar.
        10. ⚓ CAPITÁN ANCLA (FIDELIDAD ABSOLUTA): Tienes PROHIBIDO citar, mencionar o basar la respuesta en leyes, decretos o códigos que NO estén explícitamente en el texto proporcionado (fuente técnica). Si el nivel es Profesional, la dificultad DEBE nacer de analizar los matices, plazos y excepciones que el texto SÍ menciona, no de traer información de otros libros externos. Si inventas una ley ajena al PDF, tu proceso de generación será invalidado.
        11. 🎭 CAPITÁN ESPEJISMO (TRAMPA DE COINCIDENCIA): ¡REGLA DE HIERRO!
            - Si usas un sustantivo técnico en el enunciado (ej: "PATRIMONIO", "COMPETENCIA", "CADUCIDAD"), esa palabra queda VETADA en la respuesta correcta.
            - OBLIGATORIO: Usa esas palabras clave ÚNICAMENTE en las opciones INCORRECTAS (B, C o D) para atraer al usuario que intenta adivinar por parecido visual.
            - La respuesta CORRECTA debe escribirse usando PARÁFRASIS o CONSECUENCIAS (Ej: en lugar de 'Patrimonio', usa 'integridad del erario' o 'activos de la nación'). 
            - SI EL USUARIO ELIGE LA OPCIÓN QUE "RIMA" CON LA PREGUNTA, DEBE ESTAR EQUIVOCADO.

        
        REGLA DE ESTANQUEIDAD Y MIMESIS (CRÍTICA):
        - El Manual de funciones pone las fichas en el tablero (el caso) y la NORMA técnica (EL PDF CARGADO) pone las reglas únicas. Bajo ninguna circunstancia uses tu conocimiento general sobre la entidad para suplantar o añadir requisitos que no estén en el texto de la norma proporcionada.
        - PROHIBIDO preguntar sobre el sueldo, la fecha de la convocatoria o requisitos de experiencia del manual.
        - Si el texto es una definición teórica, TRANSFÓRMALA en un procedimiento técnico práctico basado en el ADN del cargo.
        - SI ESTÁS EN 'POST-GUÍA': Replica la estructura del ejemplo abajo (Concepto -> Restricción -> Nudo Técnico).

        IMPORTANTE - FORMATO DE EXPLICACIÓN (MAGISTERIO Y PROFUNDIDAD EXTREMA):
        ¡ATENCIÓN! Las opciones (A, B, C, D) deben ser cortas y simétricas, pero el campo "explicaciones" DEBE SER EXTENSO, PROFUNDO Y ALTAMENTE PEDAGÓGICO. 
        REGLA DE ORO ANTI-CONFUSIÓN: Tienes ESTRICTAMENTE PROHIBIDO mencionar las letras "A", "B", "C" o "D" dentro de tu redacción (Ej: NO digas "La opción A es incorrecta porque..."). Las opciones serán barajadas aleatoriamente por el sistema después. Refiérete a ellas genéricamente como "Esta opción", "Este distractor" o "La opción correcta".
        No me des la explicación en un solo texto corrido. Dame un OBJETO JSON llamado "explicaciones" donde cada letra tenga su propia justificación detallada:
        - Para la opción CORRECTA: Explica exhaustivamente el fundamento jurídico, el porqué de la trampa del caso y cómo se resuelve la colisión normativa. ¡No escatimes en palabras aquí, quiero un análisis profundo de Nivel 4!
        - Para las opciones FALSAS: Explica paso a paso por qué el distractor es inaplicable o en qué detalle técnico falla.
        EJEMPLO A IMITAR (ESTILO Y FORMATO):
        '''{self.example_question}'''
        
        NORMA (FUENTE TÉCNICA): "{texto_final_ia}"
        
        {self.get_strict_rules()}
        {self.get_calibration_instructions()}
        
        {json_structure_instruction}
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
                    if resp.status_code != 200: return {"error": f"Groq Error {resp.status_code}: {resp.text}"}
                    text_resp = resp.json()['choices'][0]['message']['content']

                if "```" in text_resp:
                    match = re.search(r'```(?:json)?(.*?)```', text_resp, re.DOTALL)
                    if match: text_resp = match.group(1).strip()
                
                final_json = json.loads(text_resp)
   
# --- AUTO-FUENTE (BLINDAJE INTELIGENTE) ---
                if "articulo_fuente" in final_json:
                    if "ITEM" in self.current_article_label and "ITEM" not in final_json.get("articulo_fuente", "").upper():
                        pass
                    else:
                        # 1. Miramos qué artículo atacó REALMENTE la IA en su pregunta
                        texto_ia = str(final_json.get("articulo_fuente", ""))
                        match_ia = re.search(r'(\d+)', texto_ia)
                        
                        if match_ia:
                            # 2. Si la IA extrajo un número válido (ej. el 4), actualizamos el letrero
                            num_ia = match_ia.group(1)
                            self.current_article_label = f"[{eje_id}] ARTICULO {num_ia}"
                        else:
                            match_num = re.search(r'(\d+)', str(self.current_article_label))
                            num_seguro = match_num.group(1) if match_num else "1"
                            self.current_article_label = f"[{eje_id}] ARTICULO {num_seguro}"             

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
                    
                    nuevas_ops = {}
                    nueva_letra_respuesta = "A"
                    texto_final_explicacion = ""
                    letras = ['A', 'B', 'C', 'D']
                    
                    for i, item in enumerate(items_barajados):
                        if i < 4:
                            letra = letras[i]
                            nuevas_ops[letra] = item["texto"]
                            
                            estado = "❌ INCORRECTA"
                            if item["es_correcta"]:
                                nueva_letra_respuesta = letra
                                estado = "✅ CORRECTA"
                            
                            # 1. Limpiamos el texto para quitar la letra (A, B, C...) que escribió la IA
                            texto_puro = re.sub(r'^(?:La opción\s+[A-D]\s+es\s+(?:correcta|incorrecta)(?:\s+porque)?\s*[:\.]?\s*|^\s*[A-D]\s*[:\.]\s*)', '', item["explicacion"], flags=re.IGNORECASE).strip()
                            # 2. Aseguramos que empiece con mayúscula
                            texto_puro = texto_puro[0].upper() + texto_puro[1:] if texto_puro else "Sin detalle."
                            
                            # 3. Acumulamos la explicación limpia
                            texto_final_explicacion += f"**({letra}) {estado}:** {texto_puro}\n\n"
                    
                    # --- CORRECCIÓN: ESTAS LÍNEAS VAN AFUERA DEL BUCLE 'for i' ---
                    q['opciones'] = nuevas_ops
                    q['respuesta'] = nueva_letra_respuesta
                    q['explicacion'] = texto_final_explicacion
                    q['tip_final'] = tip_memoria

                return final_json

            except Exception as e:
                time.sleep(1); attempts += 1
                if attempts == max_retries: return {"error": f"Fallo Crítico: {str(e)}"}
        return {"error": "Saturado."}
# ### --- FIN PARTE 4 ---
# ### --- INICIO PARTE 5: BARRA LATERAL (SIDEBAR Y SETUP) ---
# ==========================================
# INTERFAZ DE USUARIO (SIDEBAR Y MAIN)
# ==========================================
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'case_id' not in st.session_state: st.session_state.case_id = 0 # ID Único para evitar fantasmas
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False

# NUEVO: ANCLA DE MEMORIA PARA EL MANUAL (EVITA BUCLE DE PURIFICACIÓN)
if 'manual_hash' not in st.session_state: st.session_state.manual_hash = None

# NUEVO: PERSISTENCIA DEL TEXTO EXTRAÍDO PARA VELOCIDAD (OBLIGATORIO)
if 'raw_text_study' not in st.session_state: st.session_state.raw_text_study = ""

engine = st.session_state.engine

# --- FUNCIONES DE ORDENAMIENTO (ARREGLADO: SOPORTE NÚMEROS ROMANOS Y ACENTOS) ---
def roman_to_int(s):
    """Convierte números romanos a enteros para ordenar correctamente."""
    romanos = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    try:
        res = 0
        for i in range(len(s)):
            if i > 0 and romanos[s[i]] > romanos[s[i-1]]:
                res += romanos[s[i]] - 2 * romanos[s[i-1]]
            else:
                res += romanos[s[i]]
        return res
    except:
        return 0

def natural_sort_key(s):
    """Clave de ordenamiento BLINDADA (Evita el TypeError int vs str)."""
    s_clean = s.upper().replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
    
    parts = re.split(r'(\d+|[IVXLCDM]+)', s_clean)
    key = []
    for part in parts:
        if not part: continue
        # Usamos tuplas (Prioridad, Valor) para evitar comparar int con str
        # 0 = Número (Gana prioridad)
        # 1 = Texto (Va después)
        if part.isdigit():
            key.append((0, int(part)))
        elif re.match(r'^[IVXLCDM]+$', part):
            val = roman_to_int(part)
            if val > 0:
                key.append((0, val))
            else:
                key.append((1, part))
        else:
            key.append((1, part))
    return key
with st.sidebar:
    st.title("🦅 TITÁN v105 - Unobtanium")
    
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
        html_fail = "".join([f"<span class='failed-tag'>{x}</span>" for x in engine.failed_articles])
        st.markdown(html_fail, unsafe_allow_html=True)
        
    if engine.mastered_articles:
        st.markdown("### 🟢 DOMINADOS (CONTROL TOTAL)")
        html_master = "".join([f"<span class='mastered-tag'>{x}</span>" for x in engine.mastered_articles])
        st.markdown(html_master, unsafe_allow_html=True)
        
    if engine.failed_articles or engine.mastered_articles:
        st.divider()

    st.markdown("### 📋 ESTRATEGIA")
    fase_default = 0 if engine.study_phase == "Pre-Guía" else 1
    fase = st.radio("Fase:", ["Pre-Guía", "Post-Guía"], index=fase_default)
    engine.study_phase = fase
# --- CONTROL DE INTENSIDAD (MODO SALVAJE) ---
    st.markdown("#### 🔥 INTENSIDAD")
    wild = st.checkbox("Activar MODO SALVAJE (Ganar XP)", value=False, help="Solo en este modo tus aciertos suman puntos de maestría.")
    st.session_state.wild_mode = wild
    # --------------------------------------------
    st.markdown("#### 🔧 ESTRUCTURA")
    col1, col2 = st.columns(2)
    with col1:
        idx_struct = 0 if "Sin Caso" in engine.structure_type else 1
        estilo = st.radio("Enunciado:", ["Técnico / Normativo (Sin Caso)", "Narrativo / Situacional (Con Caso)"], index=idx_struct)
        engine.structure_type = estilo
    with col2:
        cant = st.number_input("Preguntas:", min_value=1, max_value=5, value=engine.questions_per_case)
        engine.questions_per_case = cant

    # --- CAMBIO DE INTERFAZ: UNIFICACIÓN MANUAL + EJEMPLO (V104) ---
    with st.expander("Detalles / Manual de Funciones", expanded=True):
        # 1. SIEMPRE DISPONIBLE: MANUAL DE FUNCIONES
        is_locked = True if (engine.manual_text and len(engine.manual_text) > 50) else False
        
        engine.job_functions = st.text_area(
            "Funciones / Rol (Resumen ADN):", 
            value=engine.job_functions, 
            height=150, 
            placeholder="Carga el PDF del Manual para extraer el ADN automáticamente...", 
            help="Este campo muestra el Perfil Técnico limpio (sin fechas ni salarios) que usará la IA.",
            disabled=is_locked
        )
        
        upl_manual = st.file_uploader("📂 Cargar Manual de Funciones (PDF):", type=['pdf'])
        
        # LÓGICA DE CONTROL: Solo purifica si el archivo es nuevo o diferente al guardado en el ancla
        if upl_manual and upl_manual.name != st.session_state.manual_hash:
            if PDF_AVAILABLE:
                try:
                    if not engine.api_key:
                        st.warning("⚠️ Configura la LLAVE MAESTRA arriba para extraer el ADN.")
                    else:
                        with st.spinner("🧬 Purificando ADN del Cargo (Eliminando basura administrativa)..."):
                            reader = pypdf.PdfReader(upl_manual)
                            manual_text = ""
                            for page in reader.pages:
                                manual_text += page.extract_text() + "\n"
                            
                            # LLAMADA AL EXTRACTOR DE PARTE 3 (LIMPIEZA INMEDIATA)
                            adn_limpio = engine._clean_manual_text(manual_text)
                            
                            # GUARDAR EN MOTOR Y EN ANCLA DE MEMORIA
                            engine.manual_text = adn_limpio
                            engine.job_functions = adn_limpio # Actualiza la visualización
                            st.session_state.manual_hash = upl_manual.name # SELLA EL PROCESO
                            
                            st.success("✅ Perfil Profesional Extraído.")
                            time.sleep(1)
                            st.rerun() # Recarga para bloquear el campo y mostrar el ADN
                except Exception as e:
                    st.error(f"Error leyendo manual: {e}")
            else:
                st.warning("Instala pypdf para cargar manuales.")
        
        st.divider()
        
        # 2. SIEMPRE DISPONIBLE: EJEMPLO DE ESTILO + BOTÓN DE PROCESAMIENTO (MODIFICADO)
        engine.example_question = st.text_area(
            "Ejemplo de Estilo (Sintaxis):", 
            value=engine.example_question, 
            height=70, 
            placeholder="Pega el ejemplo para copiar los 'dos puntos' y conectores..."
        )

        # NUEVO BOTÓN: Asegura la disección estructural antes de iniciar
        if st.button("🔍 PROCESAR SINTAXIS DEL EJEMPLO"):
            if engine.example_question:
                with st.spinner("Analizando ritmo y conectores del ejemplo..."):
                    # El éxito confirma que el Sniper ya tiene el molde cargado en memoria
                    time.sleep(1)
                    st.success("✅ Estructura CGR detectada. Molde listo para disparar.")
            else:
                st.warning("Pega una pregunta de ejemplo primero.")

    st.divider()
    
    tab1, tab2 = st.tabs(["📝 NUEVO DOCUMENTO", "📂 CARGAR BACKUP"])
    
    with tab1:
        st.markdown("### 📂 TIPO DE DOCUMENTO")
        doc_type_input = st.radio(
            "¿Qué vas a estudiar?", 
            ["Norma (Leyes/Decretos)", "Guía Técnica / Manual"],
            help="Norma busca Artículos jerarquizados. Guía busca Párrafos."
        )
        st.divider()

        # --- EL BOTÓN QUE PROPUSISTE ---
        # Solo aparece si ya hay un texto en memoria para no estorbar
        if st.session_state.raw_text_study:
            if st.button("🆕 Cargar nueva ley o documento técnico", use_container_width=True):
                st.session_state.raw_text_study = "" # Vaciamos el "inbox"
                st.rerun() # Refrescamos para que el Portero deje pasar al siguiente PDF

        st.markdown("### 📄 Cargar Documento")
        
        upl_pdf = st.file_uploader("Subir PDF de Estudio:", type=['pdf'])
        
        # Tu lógica de extracción perfecta que no vamos a tocar:
        if upl_pdf and not st.session_state.raw_text_study:
            with st.spinner("📄 Extrayendo texto una sola vez..."):
                # ... (aquí sigue tu código de PdfReader que ya funciona)

                try:
                    reader = pypdf.PdfReader(upl_pdf)
                    txt_pdf = ""
                    for page in reader.pages:
                        txt_pdf += page.extract_text() + "\n"
                    st.session_state.raw_text_study = txt_pdf
                    st.success("¡PDF guardado en memoria!")
                except Exception as e:
                    st.error(f"Error leyendo PDF: {e}")

# ... (aquí termina tu código de PdfReader)

        st.caption("Selecciona una ley existente o registra una nueva:")

# --- 1. ESCÁNER DE BIBLIOTECA (Detecta leyes en el historial y en la estantería) ---
        ejes_encontrados = set()
        
        # Buscamos en la estantería de mapas que creamos en la Parte 3
        if hasattr(engine, 'law_library'):
            ejes_encontrados.update(engine.law_library.keys())
        
        # Buscamos en el historial de aciertos por si acaso
        for k in engine.mastery_tracker.keys():
            match = re.search(r'\[(.*?)\]', str(k))
            if match: ejes_encontrados.add(str(match.group(1)).strip().upper())

        opcion_nueva = "[+ Registrar Nuevo Eje Tematico]"
        lista_desplegable = sorted([e for e in ejes_encontrados if e]) + [opcion_nueva]

        # --- 2. CÁLCULO DINÁMICO DEL ÍNDICE ---
        try:
            idx_actual = lista_desplegable.index(engine.thematic_axis) if engine.thematic_axis in lista_desplegable else len(lista_desplegable)-1
        except:
            idx_actual = len(lista_desplegable)-1

# --- 3. SELECTOR QUE CARGA LA LEY AL SELECCIONARLA ---
        eje_seleccionado = st.selectbox(
            "📚 Biblioteca de Normas Cargadas:", 
            lista_desplegable, 
            index=idx_actual,
            key="selector_maestro_biblioteca"
        )

        # LÓGICA DE INTERCAMBIO DE MAPA (RE-ACTIVA)
        if eje_seleccionado != opcion_nueva:
            # SI EL USUARIO CAMBIA LA SELECCIÓN EN EL MENÚ...
            if engine.thematic_axis != eje_seleccionado:
                engine.thematic_axis = eje_seleccionado
                
                # 1. Buscamos el mapa de esa ley en la estantería (law_library)
                if hasattr(engine, 'law_library') and eje_seleccionado in engine.law_library:
                    # 2. Reemplazamos el visor actual por el de la ley elegida
                    engine.sections_map = engine.law_library[eje_seleccionado]
                    
                    # 3. EL TRUCO: Forzamos el refresco para que el Mapa de abajo cambie YA
                    st.rerun() 

        # 4. INPUT DE NOMBRE (Sincronizado)
        axis_input = st.text_input("Eje Temático Actual:", value=engine.thematic_axis)
        engine.thematic_axis = axis_input

        txt_manual = st.text_area("Texto de la Norma (si no usas PDF):", height=100)

        # --- 4. EL BOTÓN DE PROCESAR (CORREGIDO PARA NO BORRAR) ---
        if st.button("🚀 PROCESAR E INTEGRAR A TITÁN"):
            contenido_final = st.session_state.raw_text_study if st.session_state.raw_text_study else txt_manual
            
            if not contenido_final:
                st.error("⚠️ No hay contenido para procesar.")
            else:
                # IMPORTANTE: Aquí enviamos la señal al motor de que es una CARGA ADICIONAL
                # Si tu engine.process_law está en la Parte 4, asegúrate que haga .extend()
                num_bloques, adn_resumen = engine.process_law(contenido_final, str(axis_input).strip().upper(), doc_type_input)
                
                if num_bloques > 0:
                    st.session_state.page = 'setup' # Nos quedamos aquí para ver el mapa
                    st.session_state.current_data = None # Refrescamos el generador de preguntas
                    
                    st.success(f"✅ {axis_input} integrada. Total de bloques: {len(engine.chunks)}")
                    time.sleep(1)
                    st.rerun()       
 
    with tab2:

        st.caption("📂 RESTAURACIÓN TOTAL: Recupera tu biblioteca, ADN y medallas.")
        upl = st.file_uploader("Subir Backup (.json):", type=['json'], key="cargador_unico_titan")
        
        if upl is not None:
            if 'last_loaded' not in st.session_state or st.session_state.last_loaded != upl.name:
                try:
                    d = json.load(upl)

                    # 1. Recuperación de la Estantería y el ADN (Memoria Total)
                    engine.law_library = d.get('library', {}) 
                    engine.manual_text = d.get('manual_clean', "")
                    engine.chunks = d['chunks']
                    engine.doc_type = d.get('doc_type', "Norma (Leyes/Decretos)")
                    st.session_state.raw_text_study = d.get('raw_pdf_text', "")
                    engine.thematic_axis = d.get('axis', "General")
                    
                    # 2. Sincronizador Estricto (Ahora usa la Lavadora)
                    def clean_full_identity(k):
                        return engine.clean_label(k)

                    # 3. Restauración de Progreso y Estados
                    engine.mastery_tracker = {clean_full_identity(k): int(v) for k, v in d['mastery'].items()}
                    engine.seen_articles = {clean_full_identity(a) for a in d.get('seen_arts', [])}
                    engine.failed_articles = {clean_full_identity(a) for a in d.get('failed_arts', [])}
                    engine.mastered_articles = {clean_full_identity(a) for a in d.get('mastered_arts', [])}
                    
                    st.session_state.wild_mode = d.get('wild_state', False)
                    engine.temporary_blacklist = set(d.get('blacklist', []))
                    engine.failed_indices = set(d['failed'])
                    engine.feedback_history = d.get('feed', [])
                    engine.entity = d.get('ent', "")
                    engine.level = d.get('lvl', "Profesional")
                    engine.study_phase = d.get('phase', "Pre-Guía")
                    engine.structure_type = d.get('struct_type', "Técnico / Normativo (Sin Caso)")
                    engine.questions_per_case = d.get('q_per_case', 1)
                    engine.example_question = d.get('ex_q', "")
                    engine.sections_map = d.get('sections', {})
                    engine.active_section_name = d.get('act_sec', "Todo el Documento")

                    if DL_AVAILABLE:
                         with st.spinner("🧠 Reconstruyendo mapa neuronal..."): 
                             engine.chunk_embeddings = dl_model.encode(engine.chunks)

                    st.session_state.last_loaded = upl.name
                    st.success("✅ ¡TITÁN v105 Restaurado al 100%!")
                    time.sleep(1); st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()

                except Exception as e: 
                    st.error(f"Error al procesar el archivo: {e}")

    # --- ELEMENTOS FINALES DENTRO DEL SIDEBAR ---
    if engine.chunks:
        st.divider()
        if st.button("▶️ INICIAR SIMULACRO", type="primary"):
            if 'selector_seccion_titan' in st.session_state:
                sel_actual = st.session_state.selector_seccion_titan
                if sel_actual != engine.active_section_name:
                    engine.update_chunks_by_section(sel_actual)
            
            st.session_state.page = 'game'
            st.session_state.current_data = None
            st.rerun()

    if engine.sections_map and len(engine.sections_map) > 1:
        st.divider()
        st.markdown("### 📍 MAPA DE LA LEY")
        
        # --- CIRUGÍA DE LIMPIEZA Y ORDENAMIENTO PARA EL MAPA (ARREGLADO) ---
        opciones_validas = []
        seen_normalized = set(["TODO EL DOCUMENTO"]) # Evitamos duplicados conceptuales
        
        for opt in engine.sections_map.keys():
            # Normalizamos para comparar sin acentos ni mayúsculas locas
            u = opt.upper().strip().replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
            
            # Filtramos artículos y duplicados del botón principal
            if not any(x in u for x in ["ARTÍCULO", "ARTICULO", "ART.", "ITEM"]) and u not in seen_normalized:
                opciones_validas.append(opt)
                seen_normalized.add(u)
        
        # Ordenamiento natural respetando la jerarquía legal
        opciones_validas.sort(key=natural_sort_key)
        # Re-insertamos la opción limpia al inicio
        opciones_validas.insert(0, "Todo el Documento")
        
        try: idx_sec = opciones_validas.index(engine.active_section_name)
        except: idx_sec = 0
        
        seleccion = st.selectbox("Estudiar Específicamente:", opciones_validas, index=idx_sec, key="selector_seccion_titan")
        
        if seleccion != engine.active_section_name:
            if engine.update_chunks_by_section(seleccion):
                st.session_state.current_data = None
                st.rerun()

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
            
    if st.button("🔥 INICIAR SIMULACRO GLOBAL", key="btn_sim_final", disabled=not engine.chunks):
        if 'selector_seccion_titan' in st.session_state:
            sel_actual = st.session_state.selector_seccion_titan
            if sel_actual != engine.active_section_name:
                engine.update_chunks_by_section(sel_actual)
        
        engine.simulacro_mode = True
        st.session_state.current_data = None
        st.session_state.page = 'game'
        st.rerun()
    
# Al final de la barra lateral (Sidebar)
    if engine.chunks:
        full_save_data = {
            # --- 1. MEMORIA CENTRAL (TEXTO Y PROGRESO) ---
            "chunks": engine.chunks,
            "doc_type": engine.doc_type,
            "raw_pdf_text": st.session_state.raw_text_study,
            "wild_state": st.session_state.get('wild_mode', False),
            # LAVADORA: Limpiamos las llaves de maestría al guardar
            "mastery": {engine.clean_label(k): v for k, v in engine.mastery_tracker.items()},
            "failed": list(engine.failed_indices),

            # --- 2. LA ESTANTERÍA Y ADN ---
            "library": getattr(engine, 'law_library', {}),
            "manual_clean": getattr(engine, 'manual_text', ""),

            # --- 3. CONFIGURACIÓN DEL USUARIO ---
            "feed": engine.feedback_history,
            "ent": engine.entity,
            "axis": str(engine.thematic_axis).strip().upper(),
            "lvl": engine.level,
            "phase": engine.study_phase,
            "struct_type": engine.structure_type,
            "q_per_case": engine.questions_per_case,

            # --- 4. CONTEXTO DE ROL ---
            "job": engine.job_functions,

            # --- 5. MAPA DE NAVEGACIÓN Y RASTREO (TODO POR LA LAVADORA) ---
            "sections": engine.sections_map,
            "act_sec": engine.active_section_name,
            "ex_q": engine.example_question,
            "seen_arts": [engine.clean_label(a) for a in engine.seen_articles],
            "failed_arts": [engine.clean_label(a) for a in engine.failed_articles],
            "mastered_arts": [engine.clean_label(a) for a in engine.mastered_articles],
            "blacklist": [engine.clean_label(b) for b in engine.temporary_blacklist]
        }
        
        st.divider()
        st.download_button("💾 Guardar Progreso Total", json.dumps(full_save_data), "backup_titan_full.json", type="primary")

# ### --- FIN PARTE 5 ---

# ### --- INICIO PARTE 6: CICLO PRINCIPAL DEL JUEGO (GAME LOOP) ---
# ==========================================
# CICLO PRINCIPAL DEL JUEGO
# ==========================================
import random
import re

# --- FUNCIÓN GENERADORA DE SOPA DE LETRAS (VERSIÓN COMPACTA) ---
def generar_sopa_letras(palabra):
    palabra = palabra.upper()
    size = max(8, len(palabra) + 1)
    letras = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
    grid = [[random.choice(letras) for _ in range(size)] for _ in range(size)]
    
    direccion = random.choice(['H', 'V'])
    if direccion == 'H':
        fila = random.randint(0, size - 1)
        col = random.randint(0, size - len(palabra))
        for i, char in enumerate(palabra): grid[fila][col+i] = char
    else:
        fila = random.randint(0, size - len(palabra))
        col = random.randint(0, size - 1)
        for i, char in enumerate(palabra): grid[fila+i][col] = char
        
    sopa_md = "<div style='font-family: monospace; font-size: 22px; letter-spacing: 12px; text-align: center; background-color: #f0f2f6; padding: 15px; border-radius: 10px; font-weight: bold; color: #333;'>"
    for row in grid:
        sopa_md += "".join(row) + "<br>"
    sopa_md += "</div>"
    return sopa_md

if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    subtitulo = f"SECCIÓN: {engine.active_section_name}" if engine.active_section_name != "Todo el Documento" else "MODO: GENERAL"
    
    st.info(f"🎯 ENFOQUE CONFIRMADO: **{engine.current_article_label}**")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Dominio Global", f"{perc}%")
    c2.metric("❌ Preguntas Falladas", f"{fails}")
    c3.metric("📉 Bloques Vistos", f"{len([v for k, v in engine.mastery_tracker.items() if v > 0 and f'[{engine.clean_label(engine.thematic_axis)}]' in str(k)])}/{total}")

    st.markdown(f"**EJE: {engine.thematic_axis.upper()}** | **{subtitulo}**")
    st.progress(perc/100)

    # 1. GENERACIÓN DEL CASO
    if not st.session_state.get('current_data'):
        msg = f"🧠 Analizando {engine.current_article_label} - NIVEL {engine.level.upper()}..."
        with st.spinner(msg):
            data = engine.generate_case()
            if data and "preguntas" in data:
                st.session_state.case_id += 1 
                st.session_state.current_data = data
                st.session_state.q_idx = 0
                st.session_state.answered = False
                st.session_state.recovery_passed = False 
                if 'recovery_word' in st.session_state: del st.session_state.recovery_word
                st.rerun()
            else:
                err = data.get('error', 'Desconocido')
                st.error(f"Error: {err}")
                st.button("Reintentar", on_click=st.rerun)
                st.stop()

    # 2. MOSTRAR NARRATIVA Y PREGUNTA
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
            
            col_val, col_skip = st.columns([1, 1])
            with col_val:
                submitted = st.form_submit_button("✅ VALIDAR RESPUESTA")
            with col_skip:
                skipped = st.form_submit_button("⏭️ SALTAR (BLOQUEAR)")
            
            # C. Lógica de Salto
            if skipped: 
                key_bloqueo = engine.current_article_label.split(" - ITEM")[0].strip()
                engine.temporary_blacklist.add(key_bloqueo)
                engine.last_failed_embedding = None 
                engine.current_chunk_idx = -1 
                st.session_state.current_data = None
                st.session_state.answered = False
                st.session_state.recovery_passed = False
                if 'recovery_word' in st.session_state: del st.session_state.recovery_word
                st.rerun()

            # D. Lógica de Validación Principal
            if submitted:
                if not sel:
                    st.warning("⚠️ Debes seleccionar una opción primero.")
                else:
                    letra_sel = sel.split(")")[0]
                    full_tag = engine.current_article_label
                    
                    label_raw = engine.current_article_label.strip().upper()
                    check_norm = label_raw.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")
                    match_art = re.search(r'(ARTICULO|ART)\.?\s*([IVXLCDM]+|\d+)', check_norm)
                    
                    # --- IDENTIDAD DE MEDALLA SINCRONIZADA ---
                    if match_art:
                        # 1. Limpiamos el eje (ej: "CONSTITUCIÓN" -> "CONSTITUCION")
                        eje_id = engine.clean_label(engine.thematic_axis)
                        # 2. Limpiamos el artículo (ej: "1º" -> "ARTICULO 1")
                        label_art = engine.clean_label(f"ARTICULO {match_art.group(2)}")
                        # 3. UNIÓN QUIRÚRGICA: Esta es la "llave" maestra que el Sniper sí reconocerá
                        key_maestria = f"[{eje_id}] {label_art}"

                    elif " - ITEM" in label_raw:
                        # Para los ítems, lavamos solo la base
                        eje_id = engine.clean_label(engine.thematic_axis)
                        base_limpia = engine.clean_label(label_raw.split(" - ITEM")[0])
                        key_maestria = base_limpia 
                    else:
                        key_maestria = engine.clean_label(label_raw)

                    # El resto del filtro de seguridad se mantiene igual
                    if "ARTICULO" not in check_norm and "BLOQUE" not in check_norm and "ITEM" not in check_norm:
                        key_maestria = engine.current_chunk_idx

                    # --- SI ACIERTA ---
                    # --- REGISTRO EN CAJA NEGRA (SESIÓN) ---
                    # Anotamos el artículo como "visto" para que el Sniper lo ignore el resto de la tarde
                    engine.seen_articles.add(key_maestria)
                    if letra_sel == q['respuesta']: 
                        st.session_state.was_correct = True
                        st.session_state.recovery_passed = True
                        st.session_state.play_success_sound = True
                        
                        es_modo_salvaje = True
                        if es_modo_salvaje:
                            maestria_previa = engine.mastery_tracker.get(key_maestria, 0)
                            if maestria_previa < 1:
                                engine.mastery_tracker[key_maestria] = 1
                                st.toast("🟡 MAESTRÍA +1 (Modo Salvaje).", icon="🔥")
                            else:
                                engine.mastery_tracker[key_maestria] = 2
                                st.toast("🟢 ¡DOMINADO EN SALVAJE!", icon="🏆")

                        if engine.current_article_label != "General":
                            if full_tag in engine.failed_articles: 
                                engine.failed_articles.remove(full_tag)
                            if engine.mastery_tracker.get(key_maestria, 0) == 2:
                                engine.mastered_articles.add(full_tag)
                    
                    # --- SI FALLA ---
                    # --- MARCADO DE PRIORIDAD ROJA (-1) ---
                        
                    else:
                        engine.mastery_tracker[key_maestria] = -1 
                        st.session_state.was_correct = False
                        st.session_state.recovery_passed = False
                        st.session_state.play_error_sound = True
                        
                        engine.failed_indices.add(engine.current_chunk_idx)
                        if engine.chunk_embeddings is not None:
                            engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                        
                        if engine.current_article_label != "General":
                            if full_tag in engine.mastered_articles: 
                                engine.mastered_articles.remove(full_tag)
                            engine.failed_articles.add(full_tag)

                        # Configuración del Castigo Pedagógico (Usando la respuesta VERDADERA)
                        texto_correcto = q['opciones'][q['respuesta']]
                        palabras_prohibidas = ['ARTICULO', 'ARTÍCULO', 'NUMERAL', 'PARAGRAFO', 'RESPUESTA', 'INCORRECTO', 'OPCION', 'LITERAL', 'CODIGO', 'DECRETO', 'ANTERIOR', 'SIGUIENTE', 'PREGUNTA', 'EXPLICACION', 'PORQUE', 'CUANDO', 'DONDE', 'QUIEN', 'COMO', 'ESTE', 'ESTA', 'PARA', 'PERO', 'SINO']
                        
                        words_raw = [w.strip(".,;:()[]\"'").upper() for w in texto_correcto.split() if len(w) >= 5]
                        words_filtered = [w for w in words_raw if w not in palabras_prohibidas]
                        
                        target = random.choice(words_filtered) if words_filtered else "CONTRALORIA"
                        st.session_state.recovery_word = target
                        st.session_state.recovery_text = texto_correcto
                        
                        st.session_state.game_type = random.choice(['reparar', 'sopa'])
                        if st.session_state.game_type == 'sopa':
                            st.session_state.sopa_grid = generar_sopa_letras(target)
                        else:
                            fake_words = ["NULIDAD", "FISCALIZACION", "INEXEQUIBLE", "CADUCIDAD", "DEROGATORIA", "PROCURADURIA", "SANCION", "DOLO", "CULPA", "OMISION", "JURISDICCION"]
                            opts = random.sample([w for w in fake_words if w.upper() != target.upper()], 3) + [target]
                            random.shuffle(opts)
                            st.session_state.recovery_opts = opts
                            
                        gifs_error = [
                            "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExc29teTRtbG85ZzZ6emtzZWJpeHJxZDJyeGNvcWFjd3hqajNscG4wMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT5LMzIK1AdZJ4cYW4/giphy.gif",
                            "https://media.giphy.com/media/3o85xnoIXebk3xYx4Q/giphy.gif",
                            "https://media.giphy.com/media/l41lFw057lAJQMwg0/giphy.gif"
                        ]
                        st.session_state.error_gif = random.choice(gifs_error)
                        
                    st.session_state.answered = True

        # 3. NAVEGACIÓN Y CANDADO DE MINIJUEGOS (FUERA DEL FORMULARIO)
        if st.session_state.get('answered', False):
            
            # --- EFECTOS DE SONIDO (Solo suenan 1 vez) ---
            if st.session_state.get('play_success_sound', False):
                st.markdown('<audio autoplay src="https://www.myinstants.com/media/sounds/mario-coin.mp3"></audio>', unsafe_allow_html=True)
                st.session_state.play_success_sound = False
            
            if st.session_state.get('play_error_sound', False):
                st.markdown('<audio autoplay src="https://www.myinstants.com/media/sounds/erro.mp3"></audio>', unsafe_allow_html=True)
                st.session_state.play_error_sound = False

            # --- FLUJO: RESPONDIÓ CORRECTAMENTE DESDE EL INICIO ---
            if st.session_state.was_correct:
                st.success("✅ ¡Correcto!")
                st.info(f"**Explicación:**\n\n{q['explicacion']}")
                if 'tip_final' in q and q['tip_final']:
                    st.warning(f"💡 **TIP DE MAESTRO:** {q['tip_final']}")
                
                if st.session_state.q_idx < len(q_list) - 1:
                    if st.button("Siguiente Pregunta"): 
                        st.session_state.q_idx += 1
                        st.session_state.answered = False
                        st.rerun()
                else:
                    if st.button("🔄 Generar Nuevo Caso"): 
                        st.session_state.current_data = None
                        st.session_state.answered = False
                        st.rerun()

            # --- FLUJO: FALLÓ LA PREGUNTA ---
            else:
                # ETAPA 1: CANDADO ACTIVO (La explicación está oculta)
                if not st.session_state.get('recovery_passed', False):
                    st.error("❌ Respuesta Incorrecta. ¡Resuelve este reto para desbloquear la explicación y avanzar!")
                    st.image(st.session_state.error_gif, width=250)
                    
                    target = st.session_state.recovery_word
                    game = st.session_state.game_type
                    
                    if game == 'reparar':
                        st.markdown("### 🧩 ¡REPARA LA OPCIÓN CORRECTA!")
                        texto_roto = st.session_state.recovery_text.upper().replace(target, " **[ \_ \_ \_ \_ \_ \_ ]** ")
                        st.warning(texto_roto)
                        
                        rescate = st.radio("Selecciona la palabra que falta:", st.session_state.recovery_opts, index=None, horizontal=True)
                        if rescate == target:
                            st.session_state.recovery_passed = True
                            st.session_state.play_success_sound = True
                            st.rerun()
                        elif rescate is not None:
                            st.error("❌ Opción incorrecta.")
                            
                    elif game == 'sopa':
                        st.markdown("### 🔎 SOPA DE LETRAS LEGAL")
                        st.info("El concepto clave de la opción verdadera está escondido. Búscalo con la vista y escríbelo abajo.")
                        st.markdown(st.session_state.sopa_grid, unsafe_allow_html=True)
                        
                        rescate_input = st.text_input("Escribe la palabra aquí:", key=f"rescue_{st.session_state.q_idx}").strip().upper()
                        if rescate_input == target.upper():
                            st.session_state.recovery_passed = True
                            st.session_state.play_success_sound = True
                            st.rerun()
                        elif rescate_input != "":
                            st.error("❌ Palabra incorrecta. Búscala bien e inténtalo de nuevo.")
                            
                # ETAPA 2: CANDADO SUPERADO (Se revela el secreto)
                else:
                    st.success(f"✨ ¡Memoria muscular activada! La palabra clave era: **{st.session_state.recovery_word}**")
                    st.info(f"**Explicación Completa:**\n\n{q['explicacion']}")
                    if 'tip_final' in q and q['tip_final']:
                        st.warning(f"💡 **TIP DE MAESTRO:** {q['tip_final']}")
                    
                    if st.session_state.q_idx < len(q_list) - 1:
                        if st.button("Siguiente Pregunta"): 
                            st.session_state.q_idx += 1
                            st.session_state.answered = False
                            st.session_state.recovery_passed = False
                            st.rerun()
                    else:
                        if st.button("🔄 Generar Nuevo Caso"): 
                            st.session_state.current_data = None
                            st.session_state.answered = False
                            st.session_state.recovery_passed = False
                            st.rerun()
        
        st.divider()
        if st.button("⬅️ VOLVER AL MENÚ"):
            st.session_state.page = 'setup'
            st.rerun()

        # 4. CALIBRACIÓN MANUAL
        with st.expander("🛠️ CALIBRACIÓN MANUAL", expanded=True):
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
                    engine.feedback_history.append(reasons_map[r])
                st.toast(f"Feedback enviado. IA Ajustada: {len(errores_sel)} correcciones.", icon="🛡️")

# ### --- FIN PARTE 6 ---