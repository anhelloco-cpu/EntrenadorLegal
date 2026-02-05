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
#  🦅 TITÁN v94: SISTEMA JURÍDICO INTEGRAL (EDICIÓN SUPREMA)
#  ----------------------------------------------------------------------------
#  ESTE CÓDIGO ES LA VERSIÓN DEFINITIVA Y COMPLETA.
#  NO BORRAR NADA. RESPETAR COMENTARIOS Y ESPACIOS.
#
#  CARACTERÍSTICAS TÉCNICAS:
#  1. MOTOR DE INTELIGENCIA SELECTIVA:
#     - Selector de Modo: El usuario define si carga "Norma" o "Guía".
#     - Segmentación Específica: Aplica reglas diferentes según el tipo.
#     - Filtro Anti-Índice: Ignora líneas de tabla de contenido (Ej: "Tema ... 5").
#
#  2. GESTIÓN DE ARCHIVOS:
#     - Lector PDF Nativo (pypdf) integrado y robusto.
#     - Procesador de Texto Manual para copias rápidas.
#     - Sistema de Backups JSON completo para guardar progreso.
#
#  3. PEDAGOGÍA Y CALIBRACIÓN:
#     - Sistema "5 Capitanes" (Calibración limpia y directa).
#     - Ordenamiento Natural (1, 2, 10...) en el menú de navegación.
#     - Barajador Inteligente de Respuestas para evitar patrones.
# ==============================================================================
# ==============================================================================


# ------------------------------------------------------------------------------
# SECCIÓN 1: GESTIÓN DE DEPENDENCIAS Y LIBRERÍAS EXTERNAS
# ------------------------------------------------------------------------------

# A. SISTEMA DE IA NEURONAL (Embeddings)
# Intentamos cargar librerías de IA avanzada para búsqueda semántica.
# Si no están presentes, el sistema usará el modo aleatorio (Fail-safe).
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
    print("✅ Cerebro Neuronal (SentenceTransformers) Activado.")
except ImportError:
    DL_AVAILABLE = False
    print("⚠️ Cerebro Neuronal no detectado. Se usará modo aleatorio.")

# B. LECTOR DE ARCHIVOS PDF (CRÍTICO PARA GUÍAS Y MANUALES)
# Intentamos cargar la librería de lectura de PDFs.
try:
    import pypdf
    PDF_AVAILABLE = True
    print("✅ Lector PDF (pypdf) Activado.")
except ImportError:
    # No forzamos la instalación automática para evitar reinicios, pero avisamos.
    PDF_AVAILABLE = False
    print("⚠️ Lector PDF no detectado. Solo se admitirá texto manual.")


# ------------------------------------------------------------------------------
# SECCIÓN 2: CONFIGURACIÓN VISUAL Y ESTILOS (TU CSS ORIGINAL COMPLETO)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TITÁN v94 - Supremo", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS avanzado para la interfaz oscura/elegante
st.markdown("""
<style>
    /* 1. Estilo para botones principales en negro elegante */
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        font-weight: bold; 
        height: 3.5em; 
        transition: all 0.3s ease-in-out; 
        background-color: #000000; 
        color: #ffffff;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 2. Caja para la narrativa del caso/norma */
    .narrative-box {
        background-color: #f8f9fa; 
        padding: 30px; 
        border-radius: 12px; 
        border-left: 6px solid #2c3e50; 
        margin-bottom: 25px;
        font-family: 'Georgia', serif; 
        font-size: 1.15em; 
        line-height: 1.6;
        color: #2c3e50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* 3. Etiquetas para artículos fallados (ROJO) */
    .failed-tag {
        background-color: #ffebee; 
        color: #c62828; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-size: 0.85em; 
        font-weight: 800; 
        margin-right: 6px;
        border: 1px solid #ef9a9a; 
        display: inline-block;
        margin-bottom: 8px;
    }

    /* 4. Etiquetas para artículos dominados (VERDE) */
    .mastered-tag {
        background-color: #e8f5e9; 
        color: #2e7d32; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-size: 0.85em; 
        font-weight: 800; 
        margin-right: 6px;
        border: 1px solid #a5d6a7; 
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* 5. Cajas estadísticas del tablero */
    .stat-box {
        text-align: center; 
        padding: 20px; 
        background: #ffffff; 
        border-radius: 12px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Ajustes generales de tipografía */
    h1, h2, h3 {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #111;
        font-weight: 600;
    }
    
    /* Ajuste para inputs de texto */
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SECCIÓN 3: CARGA DEL MODELO DE EMBEDDINGS (CACHEADO)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """
    Carga el modelo vectorial una sola vez al inicio para optimizar rendimiento.
    Esto evita recargas innecesarias cada vez que se pulsa un botón.
    """
    if DL_AVAILABLE: 
        try:
            # Usamos un modelo ligero y rápido
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            # Si falla, retornamos None y el sistema usará modo aleatorio
            return None
    return None

dl_model = load_embedding_model()


# ------------------------------------------------------------------------------
# SECCIÓN 4: LISTA MAESTRA DE ENTIDADES COLOMBIANAS
# ------------------------------------------------------------------------------
ENTIDADES_CO = [
    "Contraloría General de la República", 
    "Fiscalía General de la Nación",
    "Procuraduría General de la Nación", 
    "Defensoría del Pueblo",
    "DIAN", 
    "Registraduría Nacional del Estado Civil", 
    "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia", 
    "Consejo de Estado", 
    "Corte Constitucional",
    "Policía Nacional", 
    "Ejército Nacional", 
    "Instituto Colombiano de Bienestar Familiar (ICBF)", 
    "SENA", 
    "Ministerio de Educación Nacional", 
    "Ministerio de Salud y Protección Social", 
    "Departamento Administrativo Nacional de Estadística (DANE)",
    "Superintendencia de Industria y Comercio",
    "Superintendencia Financiera",
    "Comisión Nacional del Servicio Civil (CNSC)",
    "Otra (Manual) / Agregar +"
]


# ==============================================================================
# ==============================================================================
#  CLASE PRINCIPAL: MOTOR JURÍDICO TITÁN
#  Esta clase encapsula toda la lógica del negocio.
# ==============================================================================
# ==============================================================================
class LegalEngineTITAN:
    def __init__(self):
        # ---------------------------------------------------------
        # Variables de Almacenamiento de Datos (Estado del Sistema)
        # ---------------------------------------------------------
        self.chunks = []           # Fragmentos de texto procesado
        self.chunk_embeddings = None # Vectores matemáticos del texto
        self.mastery_tracker = {}  # Rastreador de dominio por bloque
        self.failed_indices = set() # Índices de bloques fallados
        self.feedback_history = []  # Historial de calibración (Los Capitanes)
        self.current_data = None    # Datos de la pregunta actual en pantalla
        self.current_chunk_idx = -1 # Puntero al bloque actual
        
        # ---------------------------------------------------------
        # Configuración de Usuario (Perfil)
        # ---------------------------------------------------------
        self.entity = ""
        self.level = "Profesional" 
        self.simulacro_mode = False
        self.provider = "Unknown" 
        self.api_key = ""
        self.model = None 
        self.current_temperature = 0.3 # Creatividad baja para precisión técnica
        self.last_failed_embedding = None
        self.doc_type = "Norma" # Variable CRÍTICA: Define si es Ley o Guía
        
        # ---------------------------------------------------------
        # Variables de Control Pedagógico
        # ---------------------------------------------------------
        self.study_phase = "Pre-Guía" 
        self.example_question = "" 
        self.job_functions = ""    
        self.thematic_axis = "General"
        self.structure_type = "Técnico / Normativo (Sin Caso)" 
        self.questions_per_case = 1 
        
        # ---------------------------------------------------------
        # Mapa de Documento (Índice Dinámico)
        # ---------------------------------------------------------
        self.sections_map = {} 
        self.active_section_name = "Todo el Documento"
        
        # ---------------------------------------------------------
        # Sistema Francotirador & Semáforo
        # ---------------------------------------------------------
        self.seen_articles = set()      # Artículos ya preguntados en esta sesión
        self.failed_articles = set()    # Lista Roja (Pendientes de repaso)
        self.mastered_articles = set()  # Lista Verde (Dominados)
        self.temporary_blacklist = set() # Lista Negra (Botón Saltar)
        self.current_article_label = "General"

    # --------------------------------------------------------------------------
    # MÉTODO: CONFIGURACIÓN DE API
    # Detecta automáticamente qué llave ingresó el usuario.
    # --------------------------------------------------------------------------
    def configure_api(self, key):
        key = key.strip()
        self.api_key = key
        
        if key.startswith("gsk_"):
            self.provider = "Groq"
            return True, "🚀 Motor GROQ Activado (Velocidad Sónica)"
        elif key.startswith("sk-") or key.startswith("sk-proj-"): 
            self.provider = "OpenAI"
            return True, "🤖 Motor CHATGPT (GPT-4o) Activado (Precisión Máxima)"
        else:
            self.provider = "Google"
            try:
                genai.configure(api_key=key)
                model_list = genai.list_models()
                models = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
                
                # Buscamos el mejor modelo disponible (Pro o Flash)
                target = next((m for m in models if 'gemini-1.5-pro' in m), 
                         next((m for m in models if 'flash' in m), models[0]))
                
                self.model = genai.GenerativeModel(target)
                return True, f"🧠 Motor GOOGLE ({target}) Activado"
            except Exception as e:
                return False, f"Error con la llave: {str(e)}"

    # --------------------------------------------------------------------------
    # MÉTODO: SEGMENTACIÓN INTELIGENTE SELECTIVA (EL CEREBRO DEL LECTOR v94)
    # Aquí aplicamos la lógica separada que pediste: Norma vs Guía.
    # --------------------------------------------------------------------------
    def smart_segmentation(self, full_text):
        """
        Divide el texto basándose EXCLUSIVAMENTE en el tipo de documento seleccionado.
        Esto evita que el sistema se confunda tratando de adivinar.
        """
        lineas = full_text.split('\n')
        secciones = {"Todo el Documento": []} 
        
        # Variables de estado para seguimiento de jerarquía
        active_label = None

        # --- PATRONES REGEX PARA LEYES (NORMA) ---
        p_libro = r'^\s*(LIBRO)\.?\s+[IVXLCDM]+\b'
        p_tit = r'^\s*(TÍTULO|TITULO)\.?\s+[IVXLCDM]+\b' 
        p_cap = r'^\s*(CAPÍTULO|CAPITULO)\.?\s+[IVXLCDM0-9]+\b'
        p_art = r'^\s*(ARTÍCULO|ARTICULO|ART)\.?\s*\d+'
        
        # --- PATRONES REGEX PARA GUÍAS (INDICES NUMÉRICOS) ---
        # Detecta: "1. Texto" o "10. Texto"
        p_idx_1 = r'^\s*(\d+)\.\s+([A-ZÁÉÍÓÚÑ].+)'      
        # Detecta: "1.1 Texto" o "2.3.4 Texto"
        p_idx_2 = r'^\s*(\d+\.\d+)\.?\s+([A-ZÁÉÍÓÚÑ].+)' 
        
        # --- FILTRO ANTI-ÍNDICE (EL CORTAFUEGOS) ---
        # Detecta líneas que terminan en número y tienen muchos puntos (Tabla de Contenido)
        # Ej: "5. Desarrollo ........................................... 7"
        p_basura_indice = r'\.{4,}\s*\d+\s*$' 

        for linea in lineas:
            linea_limpia = linea.strip()
            if not linea_limpia: continue
            
            # -------------------------------------------------------
            # CAMINO 1: SI ES UNA GUÍA TÉCNICA O MANUAL
            # -------------------------------------------------------
            if self.doc_type == "Guía Técnica / Manual":
                # 1. Aplicamos el Filtro Anti-Índice INMEDIATAMENTE
                # Si la línea tiene "..... 7", se muere aquí.
                if re.search(p_basura_indice, linea_limpia): 
                    continue 
                
                # 2. Buscamos Títulos Numéricos (Nivel 1)
                if re.match(p_idx_1, linea_limpia):
                    m = re.match(p_idx_1, linea_limpia)
                    active_label = f"CAPÍTULO {m.group(1)}: {m.group(2)[:80]}"
                    if active_label not in secciones: secciones[active_label] = []
                
                # 3. Buscamos Subtítulos Numéricos (Nivel 2)
                elif re.match(p_idx_2, linea_limpia):
                    m = re.match(p_idx_2, linea_limpia)
                    active_label = f"SECCIÓN {m.group(1)}: {m.group(2)[:80]}"
                    if active_label not in secciones: secciones[active_label] = []

            # -------------------------------------------------------
            # CAMINO 2: SI ES UNA NORMA (LEY, DECRETO, CÓDIGO)
            # -------------------------------------------------------
            elif self.doc_type == "Norma (Leyes/Decretos)":
                # Aquí NO aplicamos el filtro anti-índice tan agresivo.
                
                if re.match(p_libro, linea_limpia, re.I):
                    active_label = linea_limpia[:100]
                    secciones[active_label] = []
                    
                elif re.match(p_tit, linea_limpia, re.I):
                    active_label = linea_limpia[:100]
                    secciones[active_label] = []
                    
                elif re.match(p_cap, linea_limpia, re.I):
                    active_label = linea_limpia[:100]
                    secciones[active_label] = []
                
                # Nota: Los artículos se detectan para el "Francotirador", pero no crean una sección nueva
                # en el menú desplegable para no saturarlo si la ley tiene 500 artículos.

            # -------------------------------------------------------
            # GUARDADO DE DATOS (HERENCIA)
            # -------------------------------------------------------
            # El texto siempre va al "Todo el Documento"
            secciones["Todo el Documento"].append(linea)
            
            # Si hay una etiqueta activa (Capítulo, Título, etc.), guardamos la línea ahí también
            if active_label: 
                secciones[active_label].append(linea)

        # Filtramos secciones vacías o con muy poco texto (ruido)
        return {k: "\n".join(v) for k, v in secciones.items() if len(v) > 20}

    # --------------------------------------------------------------------------
    # MÉTODO: PROCESAMIENTO Y CHUNKING (DIVISIÓN)
    # --------------------------------------------------------------------------
    def process_law(self, text, axis_name, doc_type_input):
        """
        Prepara el texto para ser consumido por la IA.
        Recibe el TIPO DE DOCUMENTO del usuario.
        """
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        
        self.thematic_axis = axis_name 
        self.doc_type = doc_type_input # Guardamos la elección vital (Norma vs Guía)
        self.sections_map = self.smart_segmentation(text)
        
        # Bloques de 50.000 caracteres (Balance entre contexto y memoria)
        self.chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
        self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
        
        if dl_model: 
            with st.spinner("🧠 Generando mapa neuronal del documento..."): 
                self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(self.chunks)

    def update_chunks_by_section(self, section_name):
        """
        Permite al usuario estudiar solo una parte específica.
        """
        if section_name in self.sections_map:
            texto_seccion = self.sections_map[section_name]
            self.chunks = [texto_seccion[i:i+50000] for i in range(0, len(texto_seccion), 50000)]
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.active_section_name = section_name
            
            if dl_model: 
                self.chunk_embeddings = dl_model.encode(self.chunks)
            
            # Limpieza de memoria temporal
            self.seen_articles.clear()
            self.temporary_blacklist.clear()
            return True
        return False

    # --------------------------------------------------------------------------
    # MÉTODO: ESTADÍSTICAS
    # --------------------------------------------------------------------------
    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
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
        2. NO CHIVATEAR: No digas "Según el punto 2.1". Di "Según la guía".
        """

    # --------------------------------------------------------------------------
    # MÉTODO: GENERADOR DE CASOS (ESTRATEGIA SELECTIVA v94)
    # --------------------------------------------------------------------------
    def generate_case(self):
        """
        El cerebro de la operación. 
        Usa el TIPO DE DOCUMENTO para decidir qué buscar en el texto.
        """
        if not self.api_key: return {"error": "Falta Llave API"}
        if not self.chunks: return {"error": "Falta Documento Cargado"}
        
        # 1. Selección de Bloque (Chunk)
        idx = -1
        # Lógica de recuperación de errores (Si hay embeddings)
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]
        
        if idx == -1: idx = random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        texto_base = self.chunks[idx]
        
        # 2. ESTRATEGIA DE FRANCOTIRADOR SELECTIVA
        matches = []
        
        if self.doc_type == "Norma (Leyes/Decretos)":
            # ESTRATEGIA A: Buscar "ARTÍCULO X" (Para leyes)
            p_art = r'^\s*(?:ARTÍCULO|ARTICULO|ART)\.?\s*(\d+[A-Z]?)'
            matches = list(re.finditer(p_art, texto_base, re.I | re.M))
            
        elif self.doc_type == "Guía Técnica / Manual":
            # ESTRATEGIA B: Buscar "ÍNDICES NUMÉRICOS" (1., 1.1) (Para Guías)
            p_idx = r'^\s*(\d+\.\d+|\d+\.)\s+([A-ZÁÉÍÓÚÑ].+)'
            matches = list(re.finditer(p_idx, texto_base, re.M))

        texto_final_ia = texto_base
        self.current_article_label = "General / Sin Estructura Detectada"
        
        if matches:
            # Filtro Francotirador: Quitar lo ya visto o bloqueado
            candidatos = [m for m in matches if m.group(0).strip() not in self.seen_articles and m.group(0).strip() not in self.temporary_blacklist]
            
            if not candidatos:
                # Si se acabaron los nuevos, repetimos los no bloqueados
                candidatos = [m for m in matches if m.group(0).strip() not in self.temporary_blacklist]
                if not candidatos: 
                    # Si todo está bloqueado, reseteamos lista negra
                    candidatos = matches
                    self.temporary_blacklist.clear()
                self.seen_articles.clear()
            
            sel = random.choice(candidatos)
            start = sel.start()
            idx_m = matches.index(sel)
            
            # Cortamos hasta el siguiente elemento para aislar el tema
            end = matches[idx_m+1].start() if idx_m+1 < len(matches) else min(len(texto_base), start+4000)
            
            texto_final_ia = texto_base[start:end] 
            self.current_article_label = sel.group(0).strip()[:60] 
            
            # 3. MICRO-SEGMENTACIÓN (Universal)
            # Busca listas internas (a, b, c) o numerales internos (1, 2, 3) dentro del bloque seleccionado
            p_sub = r'(^\s*\d+\.\s+|^\s*[a-z]\)\s+|^\s*[A-Z][a-zA-Z\s]{2,50}[:\.])'
            subs = list(re.finditer(p_sub, texto_final_ia, re.M))
            
            if len(subs) > 1:
                s = random.choice(subs)
                s_start = s.start()
                s_end = subs[subs.index(s)+1].start() if subs.index(s)+1 < len(subs) else len(texto_final_ia)
                
                # Le damos contexto + el fragmento específico
                texto_final_ia = f"{texto_final_ia[:150]}\n[...]\n{texto_final_ia[s_start:s_end]}"
                self.current_article_label += f" - ITEM {s.group(0).strip()[:10]}..."
        else:
            self.current_article_label = "General"
            texto_final_ia = texto_base[:4000]

        # 4. CONFIGURACIÓN DE LOS 5 CAPITANES (CALIBRACIÓN)
        feed_instr = ""
        if self.feedback_history:
            last = self.feedback_history[-5:]
            corr = []
            
            if "pregunta_facil" in last: 
                corr.append("ALERTA CRÍTICA: El usuario se aburre. AUMENTAR DRASTICAMENTE DIFICULTAD.")
            if "respuesta_obvia" in last: 
                corr.append("ALERTA CRÍTICA: Respuestas obvias detectadas. USAR TRAMPAS LÓGICAS.")
            if "spoiler" in last: 
                corr.append("ALERTA CRÍTICA: Spoilers detectados. SIN PISTAS EN ENUNCIADO.")
            if "desconexion" in last: 
                corr.append("ALERTA CRÍTICA: Pregunta desconectada. APEGARSE AL TEXTO AL 100%.")
            if "sesgo_longitud" in last: 
                corr.append("ALERTA CRÍTICA: Patrón de longitud detectado. EQUILIBRAR OPCIONES.")
            
            if corr: feed_instr = "CORRECCIONES PRIORITARIAS DEL USUARIO: " + " ".join(corr)

        # 5. CONSTRUCCIÓN DEL PROMPT FINAL
        prompt = f"""
        ACTÚA COMO EXPERTO EN CONCURSOS (NIVEL {self.level.upper()}). 
        ENTIDAD: {self.entity.upper()}.
        TIPO DE DOCUMENTO: {self.doc_type.upper()}.
        ESTILO: {self.structure_type}.
        {feed_instr}
        
        Genera {self.questions_per_case} preguntas (A,B,C,D) basándote EXCLUSIVAMENTE en el texto proporcionado.
        
        TEXTO DE ESTUDIO:
        "{texto_final_ia}"
        
        REGLAS DE ORO:
        1. 4 OPCIONES (A,B,C,D). Una sola correcta.
        2. EXPLICACIÓN DETALLADA por opción (Por qué es correcta y por qué las otras no).
        3. TIP MEMORIA: Mnemotecnia corta o palabra clave.
        
        EJEMPLO DE ESTILO A COPIAR: 
        '''{self.example_question}'''
        
        FORMATO JSON OBLIGATORIO:
        {{
            "articulo_fuente": "REFERENCIA EXACTA (Ej: Art 5 o Punto 2.1)",
            "narrativa_caso": "Contexto situacional o normativo...",
            "preguntas": [
                {{
                    "enunciado": "...", 
                    "opciones": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }}, 
                    "respuesta": "A", 
                    "tip_memoria": "...", 
                    "explicaciones": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }}
                }}
            ]
        }}
        """
        
        # 6. LLAMADA A LA API (CON SISTEMA DE REINTENTOS)
        attempts = 0
        while attempts < 3:
            try:
                # Proveedor OpenAI
                if self.provider == "OpenAI":
                    h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    d = {"model": "gpt-4o", "messages": [{"role":"system","content":"JSON ONLY"},{"role":"user","content":prompt}], "response_format": {"type": "json_object"}}
                    r = requests.post("https://api.openai.com/v1/chat/completions", headers=h, json=d)
                    txt_resp = r.json()['choices'][0]['message']['content']
                
                # Proveedor Google
                elif self.provider == "Google":
                    res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                    txt_resp = res.text.strip()
                
                # Proveedor Groq
                else: 
                    h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    d = {"model": "llama-3.3-70b-versatile", "messages": [{"role":"system","content":"JSON ONLY"},{"role":"user","content":prompt}]}
                    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=h, json=d)
                    txt_resp = r.json()['choices'][0]['message']['content']

                # Limpieza de Markdown (Si la IA responde con ```json ... ```)
                if "```" in txt_resp: 
                    match = re.search(r'```(?:json)?(.*?)```', txt_resp, re.DOTALL)
                    if match: txt_resp = match.group(1).strip()
                
                final_json = json.loads(txt_resp)
                
                # Actualizar etiqueta si la IA detectó mejor la fuente
                if "articulo_fuente" in final_json and "ITEM" not in self.current_article_label:
                    # Si la etiqueta actual es genérica, adoptamos la de la IA
                    if "General" in self.current_article_label:
                        self.current_article_label = final_json["articulo_fuente"].upper()

                # BARAJADOR INTELIGENTE (SHUFFLE)
                # Esto evita que la respuesta correcta sea siempre la 'A' o la 'C'
                for q in final_json['preguntas']:
                    ops = list(q['opciones'].items())
                    ans_txt = q['opciones'][q['respuesta']]
                    exps = q.get('explicaciones', {})
                    
                    # Creamos objetos completos antes de barajar
                    items = [{"t":v, "e":exps.get(k,"."), "ok":(v==ans_txt)} for k,v in ops]
                    random.shuffle(items)
                    
                    new_ops = {}
                    new_ans = "A"
                    exp_txt = ""
                    lets = ['A','B','C','D']
                    
                    for i, it in enumerate(items):
                        if i < 4:
                            l = lets[i]
                            new_ops[l] = it["t"]
                            if it["ok"]: new_ans = l
                            
                            icon = "✅ CORRECTA" if it["ok"] else "❌ INCORRECTA"
                            exp_txt += f"**({l}) {icon}:** {it['e']}\n\n"
                    
                    q['opciones'] = new_ops
                    q['respuesta'] = new_ans
                    q['explicacion'] = exp_txt
                    q['tip_final'] = q.get('tip_memoria', "")
                
                return final_json
            except Exception as e: 
                time.sleep(1)
                attempts += 1
        
        return {"error": "Servidor Saturado o Error de JSON. Por favor, reintenta."}


# ==============================================================================
# ==============================================================================
#  INTERFAZ DE USUARIO (FRONTEND STREAMLIT)
#  Aquí se construye la página web visible.
# ==============================================================================
# ==============================================================================
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'case_id' not in st.session_state: st.session_state.case_id = 0
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.title("🦅 TITÁN v94 (Selectivo)")
    st.caption("Sistema de Entrenamiento Jurídico Integral")
    
    with st.expander("🔑 LLAVE MAESTRA (API KEY)", expanded=True):
        key = st.text_input("Ingresa tu Key (Google/OpenAI):", type="password")
        if key:
            ok, msg = engine.configure_api(key)
            if ok: st.success(msg)
            else: st.error(msg)
    
    st.divider()
    
    # --- SELECTOR DE MODO (NUEVO EN v94) ---
    st.markdown("### 📂 TIPO DE DOCUMENTO")
    doc_type_sel = st.radio(
        "¿Qué vas a estudiar?", 
        ["Norma (Leyes/Decretos)", "Guía Técnica / Manual"],
        help="Define cómo TITÁN leerá el archivo. Norma busca Artículos. Guía busca Numerales.",
        index=0
    )
    
    st.divider()

    # --- NAVEGACIÓN (MAPA) ---
    if engine.sections_map:
        st.markdown("### 📍 MAPA DEL DOCUMENTO")
        
        # ORDENAMIENTO NATURAL (1, 2, 10... y no 1, 10, 2)
        opciones_mapa = list(engine.sections_map.keys())
        if "Todo el Documento" in opciones_mapa: opciones_mapa.remove("Todo el Documento")
        
        def natural_keys(text):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
        
        opciones_mapa.sort(key=natural_keys)
        opciones_mapa.insert(0, "Todo el Documento")
        
        try: idx_sec = opciones_mapa.index(engine.active_section_name)
        except: idx_sec = 0
            
        sel = st.selectbox("Saltar a sección:", opciones_mapa, index=idx_sec)
        
        if sel != engine.active_section_name: 
            engine.update_chunks_by_section(sel)
            st.toast(f"Enfoque cambiado a: {sel}", icon="🗺️")
            st.rerun()

    st.divider()

    # --- PESTAÑAS DE CARGA ---
    t1, t2 = st.tabs(["📝 NUEVO DOCUMENTO", "📂 CARGAR BACKUP"])
    
    with t1:
        txt_pdf = ""
        # 1. CARGA DE PDF (INTEGRADA)
        if PDF_AVAILABLE:
            pdf = st.file_uploader("Subir PDF (Guía/Ley/Manual):", type=['pdf'])
            if pdf:
                try:
                    with st.spinner("📄 Extrayendo texto..."):
                        reader = pypdf.PdfReader(pdf)
                        for p in reader.pages: txt_pdf += p.extract_text() + "\n"
                        st.success(f"¡Leído! {len(reader.pages)} páginas.")
                except Exception as e: st.error(f"Error PDF: {e}")
        else:
            st.warning("⚠️ Librería 'pypdf' no instalada. Solo texto manual.")
        
        # 2. CARGA MANUAL
        st.caption("O pega el texto aquí:")
        txt_manual = st.text_area("Texto Manual:", height=100)
        axis = st.text_input("Tema / Eje Temático (Ej: Guía Auditoría):", value=engine.thematic_axis)
        
        if st.button("🚀 PROCESAR DOCUMENTO"):
            final = txt_pdf if txt_pdf else txt_manual
            # Pasamos el TIPO DE DOCUMENTO al procesador
            if engine.process_law(final, axis, doc_type_sel): 
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.success(f"¡Procesado como {doc_type_sel}!")
                time.sleep(1)
                st.rerun()

    with t2:
        # 3. CARGA DE BACKUP (JSON)
        upl = st.file_uploader("Subir Backup (.json):", type=['json'])
        if upl:
            try:
                d = json.load(upl)
                engine.chunks = d['chunks']
                engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
                # Recuperamos listas
                engine.failed_articles = set(d.get('failed_arts', []))
                engine.mastered_articles = set(d.get('mastered_arts', []))
                st.success("Backup Restaurado")
                time.sleep(1)
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.rerun()
            except: st.error("Archivo corrupto")
    
    # --- BOTÓN DE INICIO DE SIMULACRO ---
    if engine.chunks and st.session_state.page == 'setup':
        st.divider()
        if st.button("▶️ INICIAR ENTRENAMIENTO", type="primary"): 
            st.session_state.page = 'game'
            st.session_state.current_data = None
            st.rerun()
            
    # --- BOTÓN DE GUARDADO ---
    if engine.chunks:
        st.divider()
        # Preparamos datos para guardar
        save_data = {
            "chunks": engine.chunks,
            "mastery": engine.mastery_tracker,
            "failed_arts": list(engine.failed_articles),
            "mastered_arts": list(engine.mastered_articles)
        }
        st.download_button("💾 Guardar Progreso", json.dumps(save_data), "backup_titan.json")


# --- PANTALLA PRINCIPAL (JUEGO) ---
if st.session_state.page == 'game':
    # 1. MÉTRICAS SUPERIORES
    p, f, t = engine.get_stats()
    
    st.info(f"🎯 FOCO ACTUAL: **{engine.current_article_label}**")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Dominio Total", f"{p}%")
    c2.metric("Fallos Acumulados", f"{f}")
    c3.metric("Bloques Estudiados", f"{len([x for x in engine.mastery_tracker.values() if x>0])}/{t}")
    st.progress(p/100)

    # 2. GENERACIÓN DE PREGUNTA (Si no hay activa)
    if not st.session_state.get('current_data'):
        with st.spinner("🤖 Analizando documento y generando caso..."):
            d = engine.generate_case()
            if "error" in d: 
                st.error(d['error'])
                if st.button("Reintentar"): st.rerun()
            else: 
                st.session_state.current_data = d
                st.session_state.case_id += 1
                st.session_state.q_idx = 0
                st.session_state.answered = False
                st.rerun()

    # 3. VISUALIZACIÓN DE LA PREGUNTA
    d = st.session_state.current_data
    
    # Caja de Narrativa
    st.markdown(f"<div class='narrative-box'><h4>📜 Contexto</h4>{d.get('narrativa_caso','...')}</div>", unsafe_allow_html=True)
    
    if d.get('preguntas'):
        q_list = d['preguntas']
        if st.session_state.q_idx < len(q_list):
            q = q_list[st.session_state.q_idx]
            
            with st.form(key=f"form_{st.session_state.case_id}_{st.session_state.q_idx}"):
                st.write(f"### {q['enunciado']}")
                
                # Opciones de Radio
                # Usamos una lista de opciones pre-barajadas desde la generación
                opciones_visuales = [f"{k}) {v}" for k, v in q['opciones'].items()]
                sel = st.radio("Selecciona una opción:", opciones_visuales)
                
                c_val, c_skip = st.columns([1,1])
                submitted = c_val.form_submit_button("✅ VALIDAR RESPUESTA")
                skipped = c_skip.form_submit_button("⏭️ SALTAR TEMA (Bloquear)")
                
                # --- LÓGICA DE VALIDACIÓN ---
                if submitted:
                    if not sel:
                        st.warning("Debes seleccionar una opción.")
                    else:
                        letra_seleccionada = sel.split(")")[0]
                        if letra_seleccionada == q['respuesta']: 
                            st.success("🎉 ¡CORRECTO! Has dominado este punto.")
                            engine.mastery_tracker[engine.current_chunk_idx] += 1
                            
                            tag = f"[{engine.thematic_axis}] {engine.current_article_label}"
                            if "General" not in tag: 
                                engine.failed_articles.discard(tag)
                                engine.mastered_articles.add(tag)
                        else: 
                            st.error(f"❌ INCORRECTO. La respuesta correcta era la opción {q['respuesta']}.")
                            engine.failed_indices.add(engine.current_chunk_idx)
                            
                            # Guardamos vector de error si hay modelo
                            if engine.chunk_embeddings is not None:
                                engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                            
                            tag = f"[{engine.thematic_axis}] {engine.current_article_label}"
                            if "General" not in tag: 
                                engine.mastered_articles.discard(tag)
                                engine.failed_articles.add(tag)
                        
                        # Explicación
                        st.info(q['explicacion'])
                        
                        # Tip de Memoria
                        if q.get('tip_final'): 
                            st.warning(f"💡 **TIP DE MEMORIA:** {q['tip_final']}")
                        
                        st.session_state.answered = True
                        st.rerun()
                
                # --- LÓGICA DE SALTO ---
                if skipped:
                    # Bloqueo temporal
                    label_clean = engine.current_article_label.split(" - ")[0]
                    engine.temporary_blacklist.add(label_clean)
                    st.toast(f"Tema bloqueado por esta sesión: {label_clean}")
                    st.session_state.current_data = None
                    st.rerun()

        # 4. BOTÓN SIGUIENTE (Fuera del form para evitar recargas raras)
        if st.session_state.answered:
            col_next, col_new = st.columns(2)
            if st.session_state.q_idx < len(q_list) - 1:
                if col_next.button("Siguiente Pregunta ➡️"):
                    st.session_state.q_idx += 1
                    st.session_state.answered = False
                    st.rerun()
            else:
                if col_new.button("Finalizar Caso y Generar Nuevo 🔄"): 
                    st.session_state.current_data = None
                    st.session_state.answered = False
                    st.rerun()

    # 5. ÁREA DE CALIBRACIÓN (LOS 5 CAPITANES PUROS)
    st.divider()
    with st.expander("🛠️ CALIBRACIÓN DE IA (REPORTAR FALLOS)"):
        st.caption("Ayuda a TITÁN a mejorar. Si la pregunta fue mala, repórtalo aquí:")
        
        # SOLO LAS 5 OPCIONES CORRECTAS
        reasons_map = {
            "Muy Fácil": "pregunta_facil",
            "Respuesta Obvia": "respuesta_obvia",
            "Spoiler (Pistas en enunciado)": "spoiler",
            "Desconexión (Nada que ver)": "desconexion",
            "Opciones Desiguales (Longitud)": "sesgo_longitud"
        }
        
        errs = st.multiselect("Selecciona los fallos:", list(reasons_map.keys()))
        
        if st.button("📢 ENVIAR REPORTE Y CASTIGAR IA"):
            for e in errs: 
                engine.feedback_history.append(reasons_map[e])
            st.toast(f"Reporte enviado. La IA ha sido recalibrada con {len(errs)} castigos.", icon="🛡️")
``````python
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
#  🦅 TITÁN v94: SISTEMA JURÍDICO INTEGRAL (EDICIÓN SUPREMA)
#  ----------------------------------------------------------------------------
#  ESTE CÓDIGO ES LA VERSIÓN DEFINITIVA Y COMPLETA.
#  NO BORRAR NADA. RESPETAR COMENTARIOS Y ESPACIOS.
#
#  CARACTERÍSTICAS TÉCNICAS:
#  1. MOTOR DE INTELIGENCIA SELECTIVA:
#     - Selector de Modo: El usuario define si carga "Norma" o "Guía".
#     - Segmentación Específica: Aplica reglas diferentes según el tipo.
#     - Filtro Anti-Índice: Ignora líneas de tabla de contenido (Ej: "Tema ... 5").
#
#  2. GESTIÓN DE ARCHIVOS:
#     - Lector PDF Nativo (pypdf) integrado y robusto.
#     - Procesador de Texto Manual para copias rápidas.
#     - Sistema de Backups JSON completo para guardar progreso.
#
#  3. PEDAGOGÍA Y CALIBRACIÓN:
#     - Sistema "5 Capitanes" (Calibración limpia y directa).
#     - Ordenamiento Natural (1, 2, 10...) en el menú de navegación.
#     - Barajador Inteligente de Respuestas para evitar patrones.
# ==============================================================================
# ==============================================================================


# ------------------------------------------------------------------------------
# SECCIÓN 1: GESTIÓN DE DEPENDENCIAS Y LIBRERÍAS EXTERNAS
# ------------------------------------------------------------------------------

# A. SISTEMA DE IA NEURONAL (Embeddings)
# Intentamos cargar librerías de IA avanzada para búsqueda semántica.
# Si no están presentes, el sistema usará el modo aleatorio (Fail-safe).
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
    print("✅ Cerebro Neuronal (SentenceTransformers) Activado.")
except ImportError:
    DL_AVAILABLE = False
    print("⚠️ Cerebro Neuronal no detectado. Se usará modo aleatorio.")

# B. LECTOR DE ARCHIVOS PDF (CRÍTICO PARA GUÍAS Y MANUALES)
# Intentamos cargar la librería de lectura de PDFs.
try:
    import pypdf
    PDF_AVAILABLE = True
    print("✅ Lector PDF (pypdf) Activado.")
except ImportError:
    # No forzamos la instalación automática para evitar reinicios, pero avisamos.
    PDF_AVAILABLE = False
    print("⚠️ Lector PDF no detectado. Solo se admitirá texto manual.")


# ------------------------------------------------------------------------------
# SECCIÓN 2: CONFIGURACIÓN VISUAL Y ESTILOS (TU CSS ORIGINAL COMPLETO)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TITÁN v94 - Supremo", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS avanzado para la interfaz oscura/elegante
st.markdown("""
<style>
    /* 1. Estilo para botones principales en negro elegante */
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        font-weight: bold; 
        height: 3.5em; 
        transition: all 0.3s ease-in-out; 
        background-color: #000000; 
        color: #ffffff;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 2. Caja para la narrativa del caso/norma */
    .narrative-box {
        background-color: #f8f9fa; 
        padding: 30px; 
        border-radius: 12px; 
        border-left: 6px solid #2c3e50; 
        margin-bottom: 25px;
        font-family: 'Georgia', serif; 
        font-size: 1.15em; 
        line-height: 1.6;
        color: #2c3e50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* 3. Etiquetas para artículos fallados (ROJO) */
    .failed-tag {
        background-color: #ffebee; 
        color: #c62828; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-size: 0.85em; 
        font-weight: 800; 
        margin-right: 6px;
        border: 1px solid #ef9a9a; 
        display: inline-block;
        margin-bottom: 8px;
    }

    /* 4. Etiquetas para artículos dominados (VERDE) */
    .mastered-tag {
        background-color: #e8f5e9; 
        color: #2e7d32; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-size: 0.85em; 
        font-weight: 800; 
        margin-right: 6px;
        border: 1px solid #a5d6a7; 
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* 5. Cajas estadísticas del tablero */
    .stat-box {
        text-align: center; 
        padding: 20px; 
        background: #ffffff; 
        border-radius: 12px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Ajustes generales de tipografía */
    h1, h2, h3 {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #111;
        font-weight: 600;
    }
    
    /* Ajuste para inputs de texto */
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SECCIÓN 3: CARGA DEL MODELO DE EMBEDDINGS (CACHEADO)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """
    Carga el modelo vectorial una sola vez al inicio para optimizar rendimiento.
    Esto evita recargas innecesarias cada vez que se pulsa un botón.
    """
    if DL_AVAILABLE: 
        try:
            # Usamos un modelo ligero y rápido
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            # Si falla, retornamos None y el sistema usará modo aleatorio
            return None
    return None

dl_model = load_embedding_model()


# ------------------------------------------------------------------------------
# SECCIÓN 4: LISTA MAESTRA DE ENTIDADES COLOMBIANAS
# ------------------------------------------------------------------------------
ENTIDADES_CO = [
    "Contraloría General de la República", 
    "Fiscalía General de la Nación",
    "Procuraduría General de la Nación", 
    "Defensoría del Pueblo",
    "DIAN", 
    "Registraduría Nacional del Estado Civil", 
    "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia", 
    "Consejo de Estado", 
    "Corte Constitucional",
    "Policía Nacional", 
    "Ejército Nacional", 
    "Instituto Colombiano de Bienestar Familiar (ICBF)", 
    "SENA", 
    "Ministerio de Educación Nacional", 
    "Ministerio de Salud y Protección Social", 
    "Departamento Administrativo Nacional de Estadística (DANE)",
    "Superintendencia de Industria y Comercio",
    "Superintendencia Financiera",
    "Comisión Nacional del Servicio Civil (CNSC)",
    "Otra (Manual) / Agregar +"
]


# ==============================================================================
# ==============================================================================
#  CLASE PRINCIPAL: MOTOR JURÍDICO TITÁN
#  Esta clase encapsula toda la lógica del negocio.
# ==============================================================================
# ==============================================================================
class LegalEngineTITAN:
    def __init__(self):
        # ---------------------------------------------------------
        # Variables de Almacenamiento de Datos (Estado del Sistema)
        # ---------------------------------------------------------
        self.chunks = []           # Fragmentos de texto procesado
        self.chunk_embeddings = None # Vectores matemáticos del texto
        self.mastery_tracker = {}  # Rastreador de dominio por bloque
        self.failed_indices = set() # Índices de bloques fallados
        self.feedback_history = []  # Historial de calibración (Los Capitanes)
        self.current_data = None    # Datos de la pregunta actual en pantalla
        self.current_chunk_idx = -1 # Puntero al bloque actual
        
        # ---------------------------------------------------------
        # Configuración de Usuario (Perfil)
        # ---------------------------------------------------------
        self.entity = ""
        self.level = "Profesional" 
        self.simulacro_mode = False
        self.provider = "Unknown" 
        self.api_key = ""
        self.model = None 
        self.current_temperature = 0.3 # Creatividad baja para precisión técnica
        self.last_failed_embedding = None
        self.doc_type = "Norma" # Variable CRÍTICA: Define si es Ley o Guía
        
        # ---------------------------------------------------------
        # Variables de Control Pedagógico
        # ---------------------------------------------------------
        self.study_phase = "Pre-Guía" 
        self.example_question = "" 
        self.job_functions = ""    
        self.thematic_axis = "General"
        self.structure_type = "Técnico / Normativo (Sin Caso)" 
        self.questions_per_case = 1 
        
        # ---------------------------------------------------------
        # Mapa de Documento (Índice Dinámico)
        # ---------------------------------------------------------
        self.sections_map = {} 
        self.active_section_name = "Todo el Documento"
        
        # ---------------------------------------------------------
        # Sistema Francotirador & Semáforo
        # ---------------------------------------------------------
        self.seen_articles = set()      # Artículos ya preguntados en esta sesión
        self.failed_articles = set()    # Lista Roja (Pendientes de repaso)
        self.mastered_articles = set()  # Lista Verde (Dominados)
        self.temporary_blacklist = set() # Lista Negra (Botón Saltar)
        self.current_article_label = "General"

    # --------------------------------------------------------------------------
    # MÉTODO: CONFIGURACIÓN DE API
    # Detecta automáticamente qué llave ingresó el usuario.
    # --------------------------------------------------------------------------
    def configure_api(self, key):
        key = key.strip()
        self.api_key = key
        
        if key.startswith("gsk_"):
            self.provider = "Groq"
            return True, "🚀 Motor GROQ Activado (Velocidad Sónica)"
        elif key.startswith("sk-") or key.startswith("sk-proj-"): 
            self.provider = "OpenAI"
            return True, "🤖 Motor CHATGPT (GPT-4o) Activado (Precisión Máxima)"
        else:
            self.provider = "Google"
            try:
                genai.configure(api_key=key)
                model_list = genai.list_models()
                models = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
                
                # Buscamos el mejor modelo disponible (Pro o Flash)
                target = next((m for m in models if 'gemini-1.5-pro' in m), 
                         next((m for m in models if 'flash' in m), models[0]))
                
                self.model = genai.GenerativeModel(target)
                return True, f"🧠 Motor GOOGLE ({target}) Activado"
            except Exception as e:
                return False, f"Error con la llave: {str(e)}"

    # --------------------------------------------------------------------------
    # MÉTODO: SEGMENTACIÓN INTELIGENTE SELECTIVA (EL CEREBRO DEL LECTOR v94)
    # Aquí aplicamos la lógica separada que pediste: Norma vs Guía.
    # --------------------------------------------------------------------------
    def smart_segmentation(self, full_text):
        """
        Divide el texto basándose EXCLUSIVAMENTE en el tipo de documento seleccionado.
        Esto evita que el sistema se confunda tratando de adivinar.
        """
        lineas = full_text.split('\n')
        secciones = {"Todo el Documento": []} 
        
        # Variables de estado para seguimiento de jerarquía
        active_label = None

        # --- PATRONES REGEX PARA LEYES (NORMA) ---
        p_libro = r'^\s*(LIBRO)\.?\s+[IVXLCDM]+\b'
        p_tit = r'^\s*(TÍTULO|TITULO)\.?\s+[IVXLCDM]+\b' 
        p_cap = r'^\s*(CAPÍTULO|CAPITULO)\.?\s+[IVXLCDM0-9]+\b'
        p_art = r'^\s*(ARTÍCULO|ARTICULO|ART)\.?\s*\d+'
        
        # --- PATRONES REGEX PARA GUÍAS (INDICES NUMÉRICOS) ---
        # Detecta: "1. Texto" o "10. Texto"
        p_idx_1 = r'^\s*(\d+)\.\s+([A-ZÁÉÍÓÚÑ].+)'      
        # Detecta: "1.1 Texto" o "2.3.4 Texto"
        p_idx_2 = r'^\s*(\d+\.\d+)\.?\s+([A-ZÁÉÍÓÚÑ].+)' 
        
        # --- FILTRO ANTI-ÍNDICE (EL CORTAFUEGOS) ---
        # Detecta líneas que terminan en número y tienen muchos puntos (Tabla de Contenido)
        # Ej: "5. Desarrollo ........................................... 7"
        p_basura_indice = r'\.{4,}\s*\d+\s*$' 

        for linea in lineas:
            linea_limpia = linea.strip()
            if not linea_limpia: continue
            
            # -------------------------------------------------------
            # CAMINO 1: SI ES UNA GUÍA TÉCNICA O MANUAL
            # -------------------------------------------------------
            if self.doc_type == "Guía Técnica / Manual":
                # 1. Aplicamos el Filtro Anti-Índice INMEDIATAMENTE
                # Si la línea tiene "..... 7", se muere aquí.
                if re.search(p_basura_indice, linea_limpia): 
                    continue 
                
                # 2. Buscamos Títulos Numéricos (Nivel 1)
                if re.match(p_idx_1, linea_limpia):
                    m = re.match(p_idx_1, linea_limpia)
                    active_label = f"CAPÍTULO {m.group(1)}: {m.group(2)[:80]}"
                    if active_label not in secciones: secciones[active_label] = []
                
                # 3. Buscamos Subtítulos Numéricos (Nivel 2)
                elif re.match(p_idx_2, linea_limpia):
                    m = re.match(p_idx_2, linea_limpia)
                    active_label = f"SECCIÓN {m.group(1)}: {m.group(2)[:80]}"
                    if active_label not in secciones: secciones[active_label] = []

            # -------------------------------------------------------
            # CAMINO 2: SI ES UNA NORMA (LEY, DECRETO, CÓDIGO)
            # -------------------------------------------------------
            elif self.doc_type == "Norma (Leyes/Decretos)":
                # Aquí NO aplicamos el filtro anti-índice tan agresivo.
                
                if re.match(p_libro, linea_limpia, re.I):
                    active_label = linea_limpia[:100]
                    secciones[active_label] = []
                    
                elif re.match(p_tit, linea_limpia, re.I):
                    active_label = linea_limpia[:100]
                    secciones[active_label] = []
                    
                elif re.match(p_cap, linea_limpia, re.I):
                    active_label = linea_limpia[:100]
                    secciones[active_label] = []
                
                # Nota: Los artículos se detectan para el "Francotirador", pero no crean una sección nueva
                # en el menú desplegable para no saturarlo si la ley tiene 500 artículos.

            # -------------------------------------------------------
            # GUARDADO DE DATOS (HERENCIA)
            # -------------------------------------------------------
            # El texto siempre va al "Todo el Documento"
            secciones["Todo el Documento"].append(linea)
            
            # Si hay una etiqueta activa (Capítulo, Título, etc.), guardamos la línea ahí también
            if active_label: 
                secciones[active_label].append(linea)

        # Filtramos secciones vacías o con muy poco texto (ruido)
        return {k: "\n".join(v) for k, v in secciones.items() if len(v) > 20}

    # --------------------------------------------------------------------------
    # MÉTODO: PROCESAMIENTO Y CHUNKING (DIVISIÓN)
    # --------------------------------------------------------------------------
    def process_law(self, text, axis_name, doc_type_input):
        """
        Prepara el texto para ser consumido por la IA.
        Recibe el TIPO DE DOCUMENTO del usuario.
        """
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        
        self.thematic_axis = axis_name 
        self.doc_type = doc_type_input # Guardamos la elección vital (Norma vs Guía)
        self.sections_map = self.smart_segmentation(text)
        
        # Bloques de 50.000 caracteres (Balance entre contexto y memoria)
        self.chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
        self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
        
        if dl_model: 
            with st.spinner("🧠 Generando mapa neuronal del documento..."): 
                self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(self.chunks)

    def update_chunks_by_section(self, section_name):
        """
        Permite al usuario estudiar solo una parte específica.
        """
        if section_name in self.sections_map:
            texto_seccion = self.sections_map[section_name]
            self.chunks = [texto_seccion[i:i+50000] for i in range(0, len(texto_seccion), 50000)]
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.active_section_name = section_name
            
            if dl_model: 
                self.chunk_embeddings = dl_model.encode(self.chunks)
            
            # Limpieza de memoria temporal
            self.seen_articles.clear()
            self.temporary_blacklist.clear()
            return True
        return False

    # --------------------------------------------------------------------------
    # MÉTODO: ESTADÍSTICAS
    # --------------------------------------------------------------------------
    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks)
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
        2. NO CHIVATEAR: No digas "Según el punto 2.1". Di "Según la guía".
        """

    # --------------------------------------------------------------------------
    # MÉTODO: GENERADOR DE CASOS (ESTRATEGIA SELECTIVA v94)
    # --------------------------------------------------------------------------
    def generate_case(self):
        """
        El cerebro de la operación. 
        Usa el TIPO DE DOCUMENTO para decidir qué buscar en el texto.
        """
        if not self.api_key: return {"error": "Falta Llave API"}
        if not self.chunks: return {"error": "Falta Documento Cargado"}
        
        # 1. Selección de Bloque (Chunk)
        idx = -1
        # Lógica de recuperación de errores (Si hay embeddings)
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True)
            if candidatos: idx = candidatos[0][0]
        
        if idx == -1: idx = random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        texto_base = self.chunks[idx]
        
        # 2. ESTRATEGIA DE FRANCOTIRADOR SELECTIVA
        matches = []
        
        if self.doc_type == "Norma (Leyes/Decretos)":
            # ESTRATEGIA A: Buscar "ARTÍCULO X" (Para leyes)
            p_art = r'^\s*(?:ARTÍCULO|ARTICULO|ART)\.?\s*(\d+[A-Z]?)'
            matches = list(re.finditer(p_art, texto_base, re.I | re.M))
            
        elif self.doc_type == "Guía Técnica / Manual":
            # ESTRATEGIA B: Buscar "ÍNDICES NUMÉRICOS" (1., 1.1) (Para Guías)
            p_idx = r'^\s*(\d+\.\d+|\d+\.)\s+([A-ZÁÉÍÓÚÑ].+)'
            matches = list(re.finditer(p_idx, texto_base, re.M))

        texto_final_ia = texto_base
        self.current_article_label = "General / Sin Estructura Detectada"
        
        if matches:
            # Filtro Francotirador: Quitar lo ya visto o bloqueado
            candidatos = [m for m in matches if m.group(0).strip() not in self.seen_articles and m.group(0).strip() not in self.temporary_blacklist]
            
            if not candidatos:
                # Si se acabaron los nuevos, repetimos los no bloqueados
                candidatos = [m for m in matches if m.group(0).strip() not in self.temporary_blacklist]
                if not candidatos: 
                    # Si todo está bloqueado, reseteamos lista negra
                    candidatos = matches
                    self.temporary_blacklist.clear()
                self.seen_articles.clear()
            
            sel = random.choice(candidatos)
            start = sel.start()
            idx_m = matches.index(sel)
            
            # Cortamos hasta el siguiente elemento para aislar el tema
            end = matches[idx_m+1].start() if idx_m+1 < len(matches) else min(len(texto_base), start+4000)
            
            texto_final_ia = texto_base[start:end] 
            self.current_article_label = sel.group(0).strip()[:60] 
            
            # 3. MICRO-SEGMENTACIÓN (Universal)
            # Busca listas internas (a, b, c) o numerales internos (1, 2, 3) dentro del bloque seleccionado
            p_sub = r'(^\s*\d+\.\s+|^\s*[a-z]\)\s+|^\s*[A-Z][a-zA-Z\s]{2,50}[:\.])'
            subs = list(re.finditer(p_sub, texto_final_ia, re.M))
            
            if len(subs) > 1:
                s = random.choice(subs)
                s_start = s.start()
                s_end = subs[subs.index(s)+1].start() if subs.index(s)+1 < len(subs) else len(texto_final_ia)
                
                # Le damos contexto + el fragmento específico
                texto_final_ia = f"{texto_final_ia[:150]}\n[...]\n{texto_final_ia[s_start:s_end]}"
                self.current_article_label += f" - ITEM {s.group(0).strip()[:10]}..."
        else:
            self.current_article_label = "General"
            texto_final_ia = texto_base[:4000]

        # 4. CONFIGURACIÓN DE LOS 5 CAPITANES (CALIBRACIÓN)
        feed_instr = ""
        if self.feedback_history:
            last = self.feedback_history[-5:]
            corr = []
            
            if "pregunta_facil" in last: 
                corr.append("ALERTA CRÍTICA: El usuario se aburre. AUMENTAR DRASTICAMENTE DIFICULTAD.")
            if "respuesta_obvia" in last: 
                corr.append("ALERTA CRÍTICA: Respuestas obvias detectadas. USAR TRAMPAS LÓGICAS.")
            if "spoiler" in last: 
                corr.append("ALERTA CRÍTICA: Spoilers detectados. SIN PISTAS EN ENUNCIADO.")
            if "desconexion" in last: 
                corr.append("ALERTA CRÍTICA: Pregunta desconectada. APEGARSE AL TEXTO AL 100%.")
            if "sesgo_longitud" in last: 
                corr.append("ALERTA CRÍTICA: Patrón de longitud detectado. EQUILIBRAR OPCIONES.")
            
            if corr: feed_instr = "CORRECCIONES PRIORITARIAS DEL USUARIO: " + " ".join(corr)

        # 5. CONSTRUCCIÓN DEL PROMPT FINAL
        prompt = f"""
        ACTÚA COMO EXPERTO EN CONCURSOS (NIVEL {self.level.upper()}). 
        ENTIDAD: {self.entity.upper()}.
        TIPO DE DOCUMENTO: {self.doc_type.upper()}.
        ESTILO: {self.structure_type}.
        {feed_instr}
        
        Genera {self.questions_per_case} preguntas (A,B,C,D) basándote EXCLUSIVAMENTE en el texto proporcionado.
        
        TEXTO DE ESTUDIO:
        "{texto_final_ia}"
        
        REGLAS DE ORO:
        1. 4 OPCIONES (A,B,C,D). Una sola correcta.
        2. EXPLICACIÓN DETALLADA por opción (Por qué es correcta y por qué las otras no).
        3. TIP MEMORIA: Mnemotecnia corta o palabra clave.
        
        EJEMPLO DE ESTILO A COPIAR: 
        '''{self.example_question}'''
        
        FORMATO JSON OBLIGATORIO:
        {{
            "articulo_fuente": "REFERENCIA EXACTA (Ej: Art 5 o Punto 2.1)",
            "narrativa_caso": "Contexto situacional o normativo...",
            "preguntas": [
                {{
                    "enunciado": "...", 
                    "opciones": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }}, 
                    "respuesta": "A", 
                    "tip_memoria": "...", 
                    "explicaciones": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }}
                }}
            ]
        }}
        """
        
        # 6. LLAMADA A LA API (CON SISTEMA DE REINTENTOS)
        attempts = 0
        while attempts < 3:
            try:
                # Proveedor OpenAI
                if self.provider == "OpenAI":
                    h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    d = {"model": "gpt-4o", "messages": [{"role":"system","content":"JSON ONLY"},{"role":"user","content":prompt}], "response_format": {"type": "json_object"}}
                    r = requests.post("[https://api.openai.com/v1/chat/completions](https://api.openai.com/v1/chat/completions)", headers=h, json=d)
                    txt_resp = r.json()['choices'][0]['message']['content']
                
                # Proveedor Google
                elif self.provider == "Google":
                    res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                    txt_resp = res.text.strip()
                
                # Proveedor Groq
                else: 
                    h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    d = {"model": "llama-3.3-70b-versatile", "messages": [{"role":"system","content":"JSON ONLY"},{"role":"user","content":prompt}]}
                    r = requests.post("[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)", headers=h, json=d)
                    txt_resp = r.json()['choices'][0]['message']['content']

                # Limpieza de Markdown (Si la IA responde con ```json ... ```)
                if "```" in txt_resp: 
                    match = re.search(r'```(?:json)?(.*?)```', txt_resp, re.DOTALL)
                    if match: txt_resp = match.group(1).strip()
                
                final_json = json.loads(txt_resp)
                
                # Actualizar etiqueta si la IA detectó mejor la fuente
                if "articulo_fuente" in final_json and "ITEM" not in self.current_article_label:
                    # Si la etiqueta actual es genérica, adoptamos la de la IA
                    if "General" in self.current_article_label:
                        self.current_article_label = final_json["articulo_fuente"].upper()

                # BARAJADOR INTELIGENTE (SHUFFLE)
                # Esto evita que la respuesta correcta sea siempre la 'A' o la 'C'
                for q in final_json['preguntas']:
                    ops = list(q['opciones'].items())
                    ans_txt = q['opciones'][q['respuesta']]
                    exps = q.get('explicaciones', {})
                    
                    # Creamos objetos completos antes de barajar
                    items = [{"t":v, "e":exps.get(k,"."), "ok":(v==ans_txt)} for k,v in ops]
                    random.shuffle(items)
                    
                    new_ops = {}
                    new_ans = "A"
                    exp_txt = ""
                    lets = ['A','B','C','D']
                    
                    for i, it in enumerate(items):
                        if i < 4:
                            l = lets[i]
                            new_ops[l] = it["t"]
                            if it["ok"]: new_ans = l
                            
                            icon = "✅ CORRECTA" if it["ok"] else "❌ INCORRECTA"
                            exp_txt += f"**({l}) {icon}:** {it['e']}\n\n"
                    
                    q['opciones'] = new_ops
                    q['respuesta'] = new_ans
                    q['explicacion'] = exp_txt
                    q['tip_final'] = q.get('tip_memoria', "")
                
                return final_json
            except Exception as e: 
                time.sleep(1)
                attempts += 1
        
        return {"error": "Servidor Saturado o Error de JSON. Por favor, reintenta."}


# ==============================================================================
# ==============================================================================
#  INTERFAZ DE USUARIO (FRONTEND STREAMLIT)
#  Aquí se construye la página web visible.
# ==============================================================================
# ==============================================================================
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'case_id' not in st.session_state: st.session_state.case_id = 0
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.title("🦅 TITÁN v94 (Selectivo)")
    st.caption("Sistema de Entrenamiento Jurídico Integral")
    
    with st.expander("🔑 LLAVE MAESTRA (API KEY)", expanded=True):
        key = st.text_input("Ingresa tu Key (Google/OpenAI):", type="password")
        if key:
            ok, msg = engine.configure_api(key)
            if ok: st.success(msg)
            else: st.error(msg)
    
    st.divider()
    
    # --- SELECTOR DE MODO (NUEVO EN v94) ---
    st.markdown("### 📂 TIPO DE DOCUMENTO")
    doc_type_sel = st.radio(
        "¿Qué vas a estudiar?", 
        ["Norma (Leyes/Decretos)", "Guía Técnica / Manual"],
        help="Define cómo TITÁN leerá el archivo. Norma busca Artículos. Guía busca Numerales.",
        index=0
    )
    
    st.divider()

    # --- NAVEGACIÓN (MAPA) ---
    if engine.sections_map:
        st.markdown("### 📍 MAPA DEL DOCUMENTO")
        
        # ORDENAMIENTO NATURAL (1, 2, 10... y no 1, 10, 2)
        opciones_mapa = list(engine.sections_map.keys())
        if "Todo el Documento" in opciones_mapa: opciones_mapa.remove("Todo el Documento")
        
        def natural_keys(text):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
        
        opciones_mapa.sort(key=natural_keys)
        opciones_mapa.insert(0, "Todo el Documento")
        
        try: idx_sec = opciones_mapa.index(engine.active_section_name)
        except: idx_sec = 0
            
        sel = st.selectbox("Saltar a sección:", opciones_mapa, index=idx_sec)
        
        if sel != engine.active_section_name: 
            engine.update_chunks_by_section(sel)
            st.toast(f"Enfoque cambiado a: {sel}", icon="🗺️")
            st.rerun()

    st.divider()

    # --- PESTAÑAS DE CARGA ---
    t1, t2 = st.tabs(["📝 NUEVO DOCUMENTO", "📂 CARGAR BACKUP"])
    
    with t1:
        txt_pdf = ""
        # 1. CARGA DE PDF (INTEGRADA)
        if PDF_AVAILABLE:
            pdf = st.file_uploader("Subir PDF (Guía/Ley/Manual):", type=['pdf'])
            if pdf:
                try:
                    with st.spinner("📄 Extrayendo texto..."):
                        reader = pypdf.PdfReader(pdf)
                        for p in reader.pages: txt_pdf += p.extract_text() + "\n"
                        st.success(f"¡Leído! {len(reader.pages)} páginas.")
                except Exception as e: st.error(f"Error PDF: {e}")
        else:
            st.warning("⚠️ Librería 'pypdf' no instalada. Solo texto manual.")
        
        # 2. CARGA MANUAL
        st.caption("O pega el texto aquí:")
        txt_manual = st.text_area("Texto Manual:", height=100)
        axis = st.text_input("Tema / Eje Temático (Ej: Guía Auditoría):", value=engine.thematic_axis)
        
        if st.button("🚀 PROCESAR DOCUMENTO"):
            final = txt_pdf if txt_pdf else txt_manual
            # Pasamos el TIPO DE DOCUMENTO al procesador
            if engine.process_law(final, axis, doc_type_sel): 
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.success(f"¡Procesado como {doc_type_sel}!")
                time.sleep(1)
                st.rerun()

    with t2:
        # 3. CARGA DE BACKUP (JSON)
        upl = st.file_uploader("Subir Backup (.json):", type=['json'])
        if upl:
            try:
                d = json.load(upl)
                engine.chunks = d['chunks']
                engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
                # Recuperamos listas
                engine.failed_articles = set(d.get('failed_arts', []))
                engine.mastered_articles = set(d.get('mastered_arts', []))
                st.success("Backup Restaurado")
                time.sleep(1)
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.rerun()
            except: st.error("Archivo corrupto")
    
    # --- BOTÓN DE INICIO DE SIMULACRO ---
    if engine.chunks and st.session_state.page == 'setup':
        st.divider()
        if st.button("▶️ INICIAR ENTRENAMIENTO", type="primary"): 
            st.session_state.page = 'game'
            st.session_state.current_data = None
            st.rerun()
            
    # --- BOTÓN DE GUARDADO ---
    if engine.chunks:
        st.divider()
        # Preparamos datos para guardar
        save_data = {
            "chunks": engine.chunks,
            "mastery": engine.mastery_tracker,
            "failed_arts": list(engine.failed_articles),
            "mastered_arts": list(engine.mastered_articles)
        }
        st.download_button("💾 Guardar Progreso", json.dumps(save_data), "backup_titan.json")


# --- PANTALLA PRINCIPAL (JUEGO) ---
if st.session_state.page == 'game':
    # 1. MÉTRICAS SUPERIORES
    p, f, t = engine.get_stats()
    
    st.info(f"🎯 FOCO ACTUAL: **{engine.current_article_label}**")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Dominio Total", f"{p}%")
    c2.metric("Fallos Acumulados", f"{f}")
    c3.metric("Bloques Estudiados", f"{len([x for x in engine.mastery_tracker.values() if x>0])}/{t}")
    st.progress(p/100)

    # 2. GENERACIÓN DE PREGUNTA (Si no hay activa)
    if not st.session_state.get('current_data'):
        with st.spinner("🤖 Analizando documento y generando caso..."):
            d = engine.generate_case()
            if "error" in d: 
                st.error(d['error'])
                if st.button("Reintentar"): st.rerun()
            else: 
                st.session_state.current_data = d
                st.session_state.case_id += 1
                st.session_state.q_idx = 0
                st.session_state.answered = False
                st.rerun()

    # 3. VISUALIZACIÓN DE LA PREGUNTA
    d = st.session_state.current_data
    
    # Caja de Narrativa
    st.markdown(f"<div class='narrative-box'><h4>📜 Contexto</h4>{d.get('narrativa_caso','...')}</div>", unsafe_allow_html=True)
    
    if d.get('preguntas'):
        q_list = d['preguntas']
        if st.session_state.q_idx < len(q_list):
            q = q_list[st.session_state.q_idx]
            
            with st.form(key=f"form_{st.session_state.case_id}_{st.session_state.q_idx}"):
                st.write(f"### {q['enunciado']}")
                
                # Opciones de Radio
                # Usamos una lista de opciones pre-barajadas desde la generación
                opciones_visuales = [f"{k}) {v}" for k, v in q['opciones'].items()]
                sel = st.radio("Selecciona una opción:", opciones_visuales)
                
                c_val, c_skip = st.columns([1,1])
                submitted = c_val.form_submit_button("✅ VALIDAR RESPUESTA")
                skipped = c_skip.form_submit_button("⏭️ SALTAR TEMA (Bloquear)")
                
                # --- LÓGICA DE VALIDACIÓN ---
                if submitted:
                    if not sel:
                        st.warning("Debes seleccionar una opción.")
                    else:
                        letra_seleccionada = sel.split(")")[0]
                        if letra_seleccionada == q['respuesta']: 
                            st.success("🎉 ¡CORRECTO! Has dominado este punto.")
                            engine.mastery_tracker[engine.current_chunk_idx] += 1
                            
                            tag = f"[{engine.thematic_axis}] {engine.current_article_label}"
                            if "General" not in tag: 
                                engine.failed_articles.discard(tag)
                                engine.mastered_articles.add(tag)
                        else: 
                            st.error(f"❌ INCORRECTO. La respuesta correcta era la opción {q['respuesta']}.")
                            engine.failed_indices.add(engine.current_chunk_idx)
                            
                            # Guardamos vector de error si hay modelo
                            if engine.chunk_embeddings is not None:
                                engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                            
                            tag = f"[{engine.thematic_axis}] {engine.current_article_label}"
                            if "General" not in tag: 
                                engine.mastered_articles.discard(tag)
                                engine.failed_articles.add(tag)
                        
                        # Explicación
                        st.info(q['explicacion'])
                        
                        # Tip de Memoria
                        if q.get('tip_final'): 
                            st.warning(f"💡 **TIP DE MEMORIA:** {q['tip_final']}")
                        
                        st.session_state.answered = True
                        st.rerun()
                
                # --- LÓGICA DE SALTO ---
                if skipped:
                    # Bloqueo temporal
                    label_clean = engine.current_article_label.split(" - ")[0]
                    engine.temporary_blacklist.add(label_clean)
                    st.toast(f"Tema bloqueado por esta sesión: {label_clean}")
                    st.session_state.current_data = None
                    st.rerun()

        # 4. BOTÓN SIGUIENTE (Fuera del form para evitar recargas raras)
        if st.session_state.answered:
            col_next, col_new = st.columns(2)
            if st.session_state.q_idx < len(q_list) - 1:
                if col_next.button("Siguiente Pregunta ➡️"):
                    st.session_state.q_idx += 1
                    st.session_state.answered = False
                    st.rerun()
            else:
                if col_new.button("Finalizar Caso y Generar Nuevo 🔄"): 
                    st.session_state.current_data = None
                    st.session_state.answered = False
                    st.rerun()

    # 5. ÁREA DE CALIBRACIÓN (LOS 5 CAPITANES PUROS)
    st.divider()
    with st.expander("🛠️ CALIBRACIÓN DE IA (REPORTAR FALLOS)"):
        st.caption("Ayuda a TITÁN a mejorar. Si la pregunta fue mala, repórtalo aquí:")
        
        # SOLO LAS 5 OPCIONES CORRECTAS
        reasons_map = {
            "Muy Fácil": "pregunta_facil",
            "Respuesta Obvia": "respuesta_obvia",
            "Spoiler (Pistas en enunciado)": "spoiler",
            "Desconexión (Nada que ver)": "desconexion",
            "Opciones Desiguales (Longitud)": "sesgo_longitud"
        }
        
        errs = st.multiselect("Selecciona los fallos:", list(reasons_map.keys()))
        
        if st.button("📢 ENVIAR REPORTE Y CASTIGAR IA"):
            for e in errs: 
                engine.feedback_history.append(reasons_map[e])
            st.toast(f"Reporte enviado. La IA ha sido recalibrada con {len(errs)} castigos.", icon="🛡️")