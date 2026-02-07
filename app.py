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
#  TITÁN v104: The Sentinel
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
    page_title="TITÁN v104 - The Sentinel", 
    page_icon="⚖️", 
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
        border: 1px solid #333;
    }
    
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
        self.current_temperature = 0.3 
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
    # SEGMENTACIÓN INTELIGENTE (TITÁN V106: ARQUITECTURA HÍBRIDA + HERENCIA)
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
            p_tit = rf'^\s*(TÍTULO|TITULO|TITULO PRELIMINAR)\s*{p_word_num}?\b' 
            p_cap = rf'^\s*(CAPÍTULO|CAPITULO)\s+{p_word_num}\b'
            p_sec = rf'^\s*(SECCIÓN|SECCION)\s+{p_word_num}\b'
            # Soporte total para artículos: ARTICULO 1º, ARTÍCULO 1o., ARTICULO 1.
            p_art = r'^\s*(ARTÍCULO|ARTICULO|ART)\.?\s*(\d+[º°\.o]?|[IVXLCDM]+)\b'

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
                niveles_activos = ["TODO EL DOCUMENTO"]
                if c_libro: niveles_activos.append(c_libro)
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
            with st.spinner("🧹 Purificando Manual (Extrayendo ADN del cargo y eliminando 'basura' administrativa)..."):
                adn_summary = self._clean_manual_text(text)
                self.manual_text = adn_summary # Guardamos la versión LIMPIA para la Parte 4
                text = adn_summary # Sobrescribimos para que la segmentación también sea limpia
        else:
            # Si carga una Norma, nos aseguramos de no tener residuos de manuales anteriores
            if not hasattr(self, 'manual_text') or not self.manual_text: 
                self.manual_text = "" 
        # ---------------------------------------------------
        
        self.thematic_axis = axis_name 
        self.doc_type = doc_type_input 
        self.sections_map = self.smart_segmentation(text)
        self.active_section_name = "TODO EL DOCUMENTO"
        self.chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
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
        
        # 1. DEFINIR PATRÓN DE BÚSQUEDA
        if self.doc_type == "Norma (Leyes/Decretos)":
            p_censo = r'(?:ARTÍCULO|ARTICULO|ART)\.?\s*(?:\d+[º°\.o]?|[IVXLCDM]+)\b'
        else:
            p_censo = r'^\s*\d+(?:\.\d+)+\b' 
            
        # 2. CENSO FILTRADO (Detectar y Descartar Inexequibles)
        items_validos = []
        for match in re.finditer(p_censo, texto_estudio, re.I | re.M):
            # Miramos 200 caracteres adelante del artículo encontrado
            ventana_contexto = texto_estudio[match.end():match.end()+200].upper()
            
            # Si dice INEXEQUIBLE, DEROGADO o NULO cerca, NO LO CONTAMOS
            if "INEXEQUIBLE" in ventana_contexto or "DEROGADO" in ventana_contexto or "NULO" in ventana_contexto:
                continue
                
            items_validos.append(match.group(0).strip().upper())

        items_unicos = set(items_validos)
        
        # 3. CÁLCULO FINAL (0-1-2)
        if items_unicos:
            total = len(items_unicos)
            score = sum([min(self.mastery_tracker.get(art, 0), 2) for art in items_unicos])
        else:
            total = len(self.chunks)
            score = sum([min(v, 2) for k, v in self.mastery_tracker.items() if isinstance(k, int)])
            
        perc = int((score / (total * 2)) * 100) if total > 0 else 0
        return min(perc, 100), len(self.failed_indices), total

    def get_strict_rules(self):
        return "1. NO SPOILERS. 2. DEPENDENCIA DEL TEXTO."

    def get_calibration_instructions(self):
        return "INSTRUCCIONES: NO REPETIR TEXTO, NO 'CHIVATEAR' NIVELES."
# ### --- FIN PARTE 3 ---
# ### --- INICIO PARTE 4: EL GENERADOR DE CASOS (IA SNIPER + 6 CAPITANES) ---
    # --------------------------------------------------------------------------
    # GENERADOR DE CASOS (MODIFICADO: ANTI-PEREZA + ROL PRIORITARIO + MODO PESADILLA + 6 CAPITANES)
    # --------------------------------------------------------------------------
    def generate_case(self):
        """
        Genera la pregunta. Integra:
        1. Sniper V106 (Precisión).
        2. Semáforo (Amarillo -> Pesadilla) por IDENTIDAD.
        3. Los 6 Capitanes (Reglas de Hierro en Prompt).
        4. Filtro Anti-Inexequible.
        """
        if not self.api_key: return {"error": "Falta Llave"}
        if not self.chunks: return {"error": "Falta Norma"}
        
        idx = -1
        # Lógica de repaso de errores (Embeddings)
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            # Buscamos candidatos que no estén en Verde (Nivel 2)
            # Nota: Aquí seguimos usando índices para embeddings, pero la maestría la revisaremos por nombre luego
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 2]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]
        
        if idx == -1: idx = random.choice(range(len(self.chunks)))
        self.current_chunk_idx = idx
        
        texto_base = self.chunks[idx]
        
        # --- FILTRO 1 (CAPITÁN JUSTICIA): ESCUDO ANTI-INEXEQUIBLE (Pre-Sniper) ---
        # Si el bloque completo está muerto, lo saltamos
        if "INEXEQUIBLE" in texto_base.upper() or "DEROGADO" in texto_base.upper():
            idx = random.choice(range(len(self.chunks)))
            texto_base = self.chunks[idx]
            self.current_chunk_idx = idx

        # --- FRANCOTIRADOR SELECTIVO (SNIPER V106) ---
        matches = []
        
        if self.doc_type == "Norma (Leyes/Decretos)":
            p_art = r'^\s*(?:ARTÍCULO|ARTICULO|ART)\.?\s*(\d+[º°\.o]?|[IVXLCDM]+)\b'
            matches = list(re.finditer(p_art, texto_base, re.IGNORECASE | re.MULTILINE))
            
        elif self.doc_type == "Guía Técnica / Manual":
            p_idx = r'^\s*(\d+(?:[\.\s]\d+)*)\.?\s+(.+)'
            matches = list(re.finditer(p_idx, texto_base, re.MULTILINE))

        texto_final_ia = texto_base
        self.current_article_label = "General / Sin Estructura Detectada"
        
        if matches:
            # Filtro Francotirador + Anti-Inexequible Fino
            candidatos_validos = []
            for m in matches:
                tag = m.group(0).strip()
                # Miramos 200 chars adelante para ver si dice Inexequible
                contexto = texto_base[m.end():m.end()+200].upper()
                if "INEXEQUIBLE" in contexto or "DEROGADO" in contexto: continue
                if tag in self.seen_articles or tag in self.temporary_blacklist: continue
                candidatos_validos.append(m)

            if not candidatos_validos:
                candidatos_validos = [m for m in matches if m.group(0).strip() not in self.temporary_blacklist]
                if not candidatos_validos:
                    candidatos_validos = matches
                    self.temporary_blacklist.clear() # Reset suave
                self.seen_articles.clear()
            
            if candidatos_validos:
                seleccion = random.choice(candidatos_validos)
                start_pos = seleccion.start()
                current_match_index = matches.index(seleccion)
                
                if current_match_index + 1 < len(matches):
                    end_pos = matches[current_match_index + 1].start()
                else:
                    end_pos = min(len(texto_base), start_pos + 4000)

                texto_final_ia = texto_base[start_pos:end_pos] 
                self.current_article_label = seleccion.group(0).strip()[:60]

                # --- MICRO-SEGMENTACIÓN ---
                patron_item = r'(^\s*\d+\.\s+|^\s*[a-z]\)\s+|^\s*[A-Z][a-zA-Z\s\u00C0-\u00FF]{2,50}[:\.])'
                sub_matches = list(re.finditer(patron_item, texto_final_ia, re.MULTILINE))
                
                if len(sub_matches) > 1:
                    sel_sub = random.choice(sub_matches)
                    start_sub = sel_sub.start()
                    idx_sub = sub_matches.index(sel_sub)
                    end_sub = sub_matches[idx_sub+1].start() if idx_sub + 1 < len(sub_matches) else len(texto_final_ia)
                    
                    texto_fragmento = texto_final_ia[start_sub:end_sub]
                    id_sub = sel_sub.group(0).strip()
                    if len(id_sub) > 20: id_sub = id_sub[:20] + "..."
                    
                    encabezado = texto_final_ia[:150].split('\n')[0] 
                    texto_final_ia = f"{encabezado}\n[...]\n{texto_fragmento}"
                    self.current_article_label = f"{self.current_article_label} - ITEM {id_sub}"
        else:
            self.current_article_label = "General"
            texto_final_ia = texto_base[:4000]

        # --- CEREBRO: MODO PESADILLA (SEMÁFORO SINCRONIZADO) ---
        # Buscamos la maestría por Nombre (Identidad) para que coincida con Parte 3 y 6
        key_maestria = self.current_article_label.split(" - ITEM")[0].strip().upper()
        if "ARTÍCULO" not in key_maestria and "ITEM" not in key_maestria: key_maestria = self.current_chunk_idx
        
        maestria_actual = self.mastery_tracker.get(key_maestria, 0)
        instruccion_pesadilla = ""
        
        if maestria_actual >= 1:
            instruccion_pesadilla = """
            🔥 ALERTA MODO PESADILLA ACTIVADO (CAPITÁN PESADILLA):
            El usuario ya domina la base. PROHIBIDO hacer preguntas directas o literales.
            TU MISIÓN: Busca el parágrafo más oscuro, la excepción a la regla o un caso de frontera.
            OPCIONES: Las incorrectas deben ser 'Gemelos Legales' (conceptos que parecen correctos pero no aplican aquí).
            DIFICULTAD: 10/10.
            """

        dificultad_prompt = f"NIVEL: {self.level.upper()}."
        instruccion_estilo = "ESTILO: TÉCNICO." if "Sin Caso" in self.structure_type else "ESTILO: NARRATIVO."
        
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

        # PROMPT FINAL (CON LOS 6 CAPITANES BLINDADOS)
        prompt = f"""
        ACTÚA COMO EXPERTO EN CONCURSOS (NIVEL {self.level.upper()}).
        ENTIDAD: {self.entity.upper()}.
        TIPO DE DOCUMENTO: {self.doc_type.upper()}.
        
        {mision_entidad}
        {contexto_funcional}
        {instruccion_pesadilla}
        
        {dificultad_prompt}
        {instruccion_estilo}
        {instruccion_trampas}
        {feedback_instr}
        
        Genera {self.questions_per_case} preguntas basándote EXCLUSIVAMENTE en el texto proporcionado abajo.
        
        REGLAS DE ORO (LOS 6 CAPITANES - BLINDAJE DE ÉLITE):
        1. 🚫 CAPITÁN ANTI-LORO: PROHIBIDO iniciar la respuesta con "Según el artículo...", "De acuerdo a la ley..." o similar. La respuesta debe ser una CONSECUENCIA JURÍDICA o TÉCNICA autónoma (Ej: "Se declara la nulidad...", "Opera el silencio administrativo...").
        2. 👯 CAPITÁN GEMELOS: Las opciones incorrectas NO pueden ser absurdas. Deben ser "Gemelos Legales": conceptos reales (como plazos de otras leyes, figuras parecidas) que sean plausibles para un novato pero incorrectos en este caso específico.
        3. ⚖️ CAPITÁN ECUALIZADOR: OBLIGATORIO. Las opciones A, B, C y D deben tener una LONGITUD VISUAL IDÉNTICA. Si la correcta es larga, rellena las incorrectas. Nadie debe adivinar por el tamaño del texto.
        4. 🧠 CAPITÁN ANTI-OBVIEDAD (Descarte Imposible): PROHIBIDO usar "Todas las anteriores", "Ninguna de las anteriores" o respuestas de sentido común moral. Aplica la PRUEBA DEL 50/50: La diferencia entre la correcta y la distractor más fuerte debe ser un matiz técnico (un plazo, una competencia, una excepción).
        5. 🗑️ CAPITÁN JUSTICIA: Si el fragmento de texto contiene "INEXEQUIBLE", "DEROGADO" o "NULO", IGNÓRALO COMPLETAMENTE y busca otro parágrafo vigente. No preguntes sobre leyes muertas.
        6. 🔗 CAPITÁN CONTEXTO: La pregunta debe depender del CASO HIPOTÉTICO diseñado según el ADN técnico del cargo.
        
        REGLA DE ESTANQUEIDAD (CRÍTICA):
        - La NORMA legal abajo es la ÚNICA fuente técnica para las respuestas.
        - El MANUAL de funciones es solo un escenario narrativo. PROHIBIDO usar datos administrativos (sueldos, fechas de convocatoria) en el examen.
        - Si el texto es una definición teórica, TRANSFÓRMALA en un procedimiento técnico práctico. No preguntes memoria.
        
        OTRAS REGLAS:
        - FORMATO DE ENUNCIADO: El 'enunciado' NO debe ser una pregunta ni terminar con signos de interrogación. Debe ser una instrucción directa (ej: 'Determine la acción correcta...').
        - ANTI-PEREZA: PROHIBIDO preguntar sobre firmas, vigencias o frases de cajón.

        IMPORTANTE - FORMATO DE EXPLICACIÓN (ESTRUCTURADO):
        No me des la explicación en un solo texto corrido.
        Dame un OBJETO JSON llamado "explicaciones" donde cada letra (A, B, C, D) tenga su propia explicación individual.
        
        EJEMPLO A IMITAR (ESTILO Y FORMATO):
        '''{self.example_question}'''
        
        NORMA (FUENTE TÉCNICA): "{texto_final_ia}"
        
        {self.get_strict_rules()}
        {self.get_calibration_instructions()}
        
        FORMATO JSON OBLIGATORIO:
        {{
            "articulo_fuente": "{self.current_article_label}",
            "narrativa_caso": "Texto de contexto situacional del rol...",
            "preguntas": [
                {{
                    "enunciado": "Instrucción directa...", 
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
                            
                            texto_final_explicacion += f"**({letra}) {estado}:** {item['explicacion']}\n\n"
                    
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

# --- FUNCIONES DE ORDENAMIENTO (NUEVO: SOPORTE NÚMEROS ROMANOS) ---
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
    """Clave de ordenamiento que entiende Números y Romanos."""
    # Separa el texto en bloques de números o palabras
    parts = re.split(r'(\d+|[IVXLCDM]+)', s.upper())
    key = []
    for part in parts:
        if not part: continue
        # Si es dígito normal
        if part.isdigit():
            key.append(int(part))
        # Si parece romano (ej. "IV", "X") lo convertimos
        elif re.match(r'^[IVXLCDM]+$', part):
            val = roman_to_int(part)
            # Si la conversión da 0 o es muy raro, lo dejamos como texto
            key.append(val if val > 0 else part)
        else:
            key.append(part)
    return key

with st.sidebar:
    st.title("🦅 TITÁN v104 The Sentinel")
    
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
        # LÓGICA DE BLOQUEO: Si hay ADN extraído, se bloquea la edición manual para asegurar la integridad
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
        
        # 2. SIEMPRE DISPONIBLE: EJEMPLO DE ESTILO
        engine.example_question = st.text_area("Ejemplo de Estilo (Sintaxis):", value=engine.example_question, height=70, placeholder="Pega el ejemplo para copiar los 'dos puntos' y conectores...")

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
        
        st.markdown("### 📄 Cargar Documento")
        
        upl_pdf = st.file_uploader("Subir PDF de Estudio:", type=['pdf'])
        
        if upl_pdf and not st.session_state.raw_text_study:
            with st.spinner("📄 Extrayendo texto una sola vez..."):
                try:
                    reader = pypdf.PdfReader(upl_pdf)
                    txt_pdf = ""
                    for page in reader.pages:
                        txt_pdf += page.extract_text() + "\n"
                    st.session_state.raw_text_study = txt_pdf
                    st.success("¡PDF guardado en memoria!")
                except Exception as e:
                    st.error(f"Error leyendo PDF: {e}")

        st.caption("O pega aquí el texto manualmente:")
        axis_input = st.text_input("Eje Temático (Ej: Ley 1755):", value=engine.thematic_axis)
        txt_manual = st.text_area("Texto de la Norma:", height=150)
        
        if st.button("🚀 PROCESAR Y SEGMENTAR"):
            contenido_final = st.session_state.raw_text_study if st.session_state.raw_text_study else txt_manual
            # AHORA CAPTURAMOS EL RETORNO DUAL (Bloques, ADN si aplica)
            num_bloques, adn_resumen = engine.process_law(contenido_final, axis_input, doc_type_input)
            
            if num_bloques > 0:
                # Si procesamos un manual como documento base, actualizamos el ADN también
                if doc_type_input == "Guía Técnica / Manual" and adn_resumen:
                    engine.job_functions = adn_resumen
                
                st.session_state.current_data = None
                st.success(f"¡Documento Procesado!")
                time.sleep(0.5)
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
                    engine.seen_articles = set(d.get('seen_arts', []))
                    engine.failed_articles = set(d.get('failed_arts', []))
                    engine.mastered_articles = set(d.get('mastered_arts', []))

                    if DL_AVAILABLE:
                         with st.spinner("🧠 Recuperando memoria neuronal..."): 
                             engine.chunk_embeddings = dl_model.encode(engine.chunks)

                    st.session_state.last_loaded = upl.name
                    st.success("¡Backup Cargado!")
                    time.sleep(1); st.session_state.page = 'game'; st.session_state.current_data = None; st.rerun()
                except Exception as e: 
                    st.error(f"Error al leer: {e}")

    # --- ELEMENTOS FINALES DENTRO DEL SIDEBAR ---
    
    # 1. BOTÓN SUPERIOR BLINDADO (AHORA TAMBIÉN HACE SYNC)
    if engine.chunks:
        st.divider()
        if st.button("▶️ INICIAR SIMULACRO", type="primary"):
            # LÓGICA DE SINCRONIZACIÓN (NUEVA AQUÍ)
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
        
        # --- FILTRO DE EXCLUSIÓN PARA OCULTAR ARTÍCULOS ---
        opciones_brutas = list(engine.sections_map.keys())
        opciones = [
            opt for opt in opciones_brutas 
            if not any(x in opt.upper() for x in ["ARTÍCULO", "ARTICULO", "ART.", "ITEM"])
        ]
        
        if "Todo el Documento" in opciones: opciones.remove("Todo el Documento")
        
        # --- ORDENAMIENTO ---
        opciones.sort(key=natural_sort_key)
        opciones.insert(0, "Todo el Documento")
        
        try: idx_sec = opciones.index(engine.active_section_name)
        except: idx_sec = 0
        
        # KEY NECESARIA PARA LA SINCRONIZACIÓN
        seleccion = st.selectbox("Estudiar Específicamente:", opciones, index=idx_sec, key="selector_seccion_titan")
        
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
            
    # 2. BOTÓN INFERIOR BLINDADO
    if st.button("🔥 INICIAR SIMULACRO GLOBAL", key="btn_sim_final", disabled=not engine.chunks):
        # LÓGICA DE SINCRONIZACIÓN
        if 'selector_seccion_titan' in st.session_state:
            sel_actual = st.session_state.selector_seccion_titan
            if sel_actual != engine.active_section_name:
                engine.update_chunks_by_section(sel_actual)
        
        engine.simulacro_mode = True
        st.session_state.current_data = None
        st.session_state.page = 'game'
        st.rerun()
    
    if engine.chunks:
        full_save_data = {
            "chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices),
            "feed": engine.feedback_history, "ent": engine.entity, "axis": engine.thematic_axis,
            "lvl": engine.level, "phase": engine.study_phase, "ex_q": engine.example_question, "job": engine.job_functions,
            "struct_type": engine.structure_type, "q_per_case": engine.questions_per_case,
            "sections": engine.sections_map, "act_sec": engine.active_section_name,
            "seen_arts": list(engine.seen_articles), "failed_arts": list(engine.failed_articles), "mastered_arts": list(engine.mastered_articles)
        }
        st.download_button("💾 Guardar Progreso", json.dumps(full_save_data), "backup_titan_full.json")
# ### --- FIN PARTE 5 ---
# ### --- INICIO PARTE 6: CICLO PRINCIPAL DEL JUEGO (GAME LOOP) ---
# ==========================================
# CICLO PRINCIPAL DEL JUEGO
# ==========================================
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    subtitulo = f"SECCIÓN: {engine.active_section_name}" if engine.active_section_name != "Todo el Documento" else "MODO: GENERAL"
    
    st.info(f"🎯 ENFOQUE CONFIRMADO: **{engine.current_article_label}**")
    
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
            
            col_val, col_skip = st.columns([1, 1])
            with col_val:
                submitted = st.form_submit_button("✅ VALIDAR RESPUESTA")
            with col_skip:
                skipped = st.form_submit_button("⏭️ SALTAR (BLOQUEAR)")
            
            if skipped: 
                # Bloqueo inteligente: Usa el nombre real para la lista negra
                key_bloqueo = engine.current_article_label.split(" - ITEM")[0].strip()
                engine.temporary_blacklist.add(key_bloqueo)
                st.session_state.current_data = None; st.rerun()

            if submitted:
                if not sel:
                    st.warning("⚠️ Debes seleccionar una opción primero.")
                else:
                    letra_sel = sel.split(")")[0]
                    full_tag = f"[{engine.thematic_axis}] {engine.current_article_label}"
                    
                    # --- DEFINICIÓN DE CLAVE DE MAESTRÍA (SOLDADURA FINAL) ---
                    # Usamos el nombre del artículo para que coincida con el Censo de Parte 3
                    key_maestria = engine.current_article_label.strip().upper()
                    if " - ITEM" in key_maestria: # Si es un sub-item, sumamos al padre
                        key_maestria = key_maestria.split(" - ITEM")[0].strip()
                    
                    # Si por alguna razón es "General", usamos el índice como fallback
                    if "ARTÍCULO" not in key_maestria and "BLOQUE" not in key_maestria and "ITEM" not in key_maestria:
                         key_maestria = engine.current_chunk_idx

                    if letra_sel == q['respuesta']: 
                        st.success("✅ ¡Correcto!") 
                        
                        # --- LÓGICA DE SEMÁFORO (0->1->2) CON CLAVE DE IDENTIDAD ---
                        maestria_previa = engine.mastery_tracker.get(key_maestria, 0)
                        
                        if maestria_previa < 1:
                            # Primer acierto: Pasa a AMARILLO
                            engine.mastery_tracker[key_maestria] = 1
                            st.toast("🟡 ARTÍCULO EN AMARILLO. Siguiente: MODO PESADILLA.", icon="🟡")
                        else:
                            # Segundo acierto: Pasa a VERDE (Dominado)
                            engine.mastery_tracker[key_maestria] = 2
                            st.toast("🟢 ¡DOMINADO! Artículo en Verde.", icon="🟢")

                        if engine.current_article_label != "General":
                            if full_tag in engine.failed_articles: engine.failed_articles.remove(full_tag)
                            # Sidebar solo muestra si es Verde (Nivel 2)
                            if engine.mastery_tracker.get(key_maestria, 0) == 2:
                                engine.mastered_articles.add(full_tag)
                    else: 
                        st.error(f"Incorrecto. Era {q['respuesta']}")
                        
                        # Penalización: Guardamos el fallo en el índice para Embeddings
                        engine.failed_indices.add(engine.current_chunk_idx)
                        if engine.chunk_embeddings is not None:
                            engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                        
                        # Penalización Visual: Etiqueta Roja
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
        if st.button("⬅️ VOLVER AL MENÚ"):
            st.session_state.page = 'setup'
            st.rerun()

        # --- CALIBRACIÓN MANUAL ---
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