import streamlit as st
import google.generativeai as genai
import json
import random
import time
import requests
import re
from collections import Counter

# ==========================================
# GESTIÓN DE DEPENDENCIAS
# ==========================================
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# ==========================================
# CONFIGURACIÓN VISUAL
# ==========================================
st.set_page_config(
    page_title="TITÁN v74 - Justicia Restaurada", 
    page_icon="⚖️", 
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s; background-color: #000000; color: white;}
    .narrative-box {
        background-color: #f5f5f5; padding: 25px; border-radius: 12px; 
        border-left: 6px solid #424242; margin-bottom: 25px;
        font-family: 'Georgia', serif; font-size: 1.15em; line-height: 1.6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .failed-tag {
        background-color: #ffcccc; color: #990000; padding: 4px 8px; 
        border-radius: 4px; font-size: 0.9em; font-weight: bold; margin-right: 5px;
        border: 1px solid #cc0000; display: inline-block; margin-bottom: 5px;
    }
    .mastered-tag {
        background-color: #ccffcc; color: #006600; padding: 4px 8px; 
        border-radius: 4px; font-size: 0.9em; font-weight: bold; margin-right: 5px;
        border: 1px solid #006600; display: inline-block; margin-bottom: 5px;
    }
    .stat-box { text-align: center; padding: 10px; background: #ffffff; border-radius: 8px; border: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    if DL_AVAILABLE: return SentenceTransformer('all-MiniLM-L6-v2')
    return None

dl_model = load_embedding_model()

ENTIDADES_CO = [
    "Contraloría General de la República", "Fiscalía General de la Nación", 
    "Procuraduría General de la Nación", "Defensoría del Pueblo", "DIAN", 
    "Registraduría Nacional", "Consejo Superior de la Judicatura",
    "Corte Suprema de Justicia", "Consejo de Estado", "Corte Constitucional", 
    "Policía Nacional", "Ejército Nacional", "ICBF", "SENA", 
    "Ministerio de Educación", "Ministerio de Salud", "DANE", "Otra..."
]

# ==========================================
# MOTOR JURÍDICO
# ==========================================
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
        self.current_temperature = 0.3 
        self.last_failed_embedding = None
        self.study_phase = "Pre-Guía" 
        self.example_question = "" 
        self.job_functions = ""    
        self.thematic_axis = "General"
        self.structure_type = "Técnico / Normativo (Sin Caso)" 
        self.questions_per_case = 1 
        self.sections_map = {} 
        self.active_section_name = "Todo el Documento"
        self.seen_articles = set()    
        self.failed_articles = set()   
        self.mastered_articles = set() 
        self.temporary_blacklist = set() # Lista Negra de Sesión (v72)
        self.current_article_label = "General"

    def configure_api(self, key):
        key = key.strip(); self.api_key = key
        if key.startswith("gsk_"): self.provider = "Groq"; return True, "🚀 Groq Activado"
        elif key.startswith("sk-") or key.startswith("sk-proj-"): self.provider = "OpenAI"; return True, "🤖 GPT-4o Activado"
        else:
            self.provider = "Google"
            try:
                genai.configure(api_key=key); model_list = genai.list_models()
                models = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
                target = next((m for m in models if 'gemini-1.5-pro' in m), next((m for m in models if 'flash' in m), models[0]))
                self.model = genai.GenerativeModel(target); return True, f"🧠 Google ({target}) Activado"
            except Exception as e: return False, f"Error: {str(e)}"

    def smart_segmentation(self, full_text):
        lineas = full_text.split('\n'); secciones = {"Todo el Documento": []} 
        active_hierarchy = {"LIBRO": None, "TÍTULO": None, "CAPÍTULO": None, "SECCIÓN": None, "ARTÍCULO": None}
        
        # Regex Flexibles
        p_lib = r'^\s*(LIBRO)\.?\s+[IVXLCDM]+\b'
        p_tit_rom = r'^\s*([IVXLCDM]+)\.\s+(.+)' 
        p_tit_txt = r'^\s*(TÍTULO|TITULO)\.?\s+[IVXLCDM]+\b' 
        p_cap = r'^\s*(CAPÍTULO|CAPITULO)\.?\s+[IVXLCDM0-9]+\b'
        p_sec = r'^\s*(SECCIÓN|SECCION)\.?\s+'
        p_art = r'^\s*(ARTÍCULO|ARTICULO|ART)\.?\s*\d+'

        def buscar_continuacion(idx_actual):
            for i in range(1, 4): 
                if idx_actual + i < len(lineas):
                    txt = lineas[idx_actual + i].strip()
                    if txt:
                        if re.match(r'^(ART|CAP|TIT|LIB|SEC)', txt, re.IGNORECASE): return None
                        return txt
            return None

        for idx, linea in enumerate(lineas):
            linea_limpia = linea.strip(); 
            if not linea_limpia: continue
            
            if re.match(p_lib, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60:
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                active_hierarchy["LIBRO"] = label; active_hierarchy["TÍTULO"] = None; active_hierarchy["CAPÍTULO"] = None; active_hierarchy["SECCIÓN"] = None; secciones[label] = []

            elif re.match(p_tit_rom, linea_limpia, re.IGNORECASE) or re.match(p_tit_txt, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60:
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                active_hierarchy["TÍTULO"] = label; active_hierarchy["CAPÍTULO"] = None; active_hierarchy["SECCIÓN"] = None; secciones[label] = []

            elif re.match(p_cap, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60:
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                active_hierarchy["CAPÍTULO"] = label; active_hierarchy["SECCIÓN"] = None; secciones[label] = []

            elif re.match(p_sec, linea_limpia, re.IGNORECASE):
                label = linea_limpia[:100]
                if len(label) < 60:
                    extra = buscar_continuacion(idx)
                    if extra: label = f"{label} - {extra}"
                active_hierarchy["SECCIÓN"] = label; secciones[label] = []
            
            elif re.match(p_art, linea_limpia, re.IGNORECASE):
                label = linea_limpia.split('.')[0] + "."; active_hierarchy["ARTÍCULO"] = label

            secciones["Todo el Documento"].append(linea) 
            for k in ["LIBRO", "TÍTULO", "CAPÍTULO", "SECCIÓN"]:
                if active_hierarchy[k]: secciones[active_hierarchy[k]].append(linea)
        return {k: "\n".join(v) for k, v in secciones.items() if v}

    def process_law(self, text, axis_name):
        text = text.replace('\r', ''); self.thematic_axis = axis_name 
        self.sections_map = self.smart_segmentation(text)
        self.chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
        self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
        if dl_model: 
            with st.spinner("🧠 Analizando jerarquía..."): self.chunk_embeddings = dl_model.encode(self.chunks)
        return len(self.chunks)

    def update_chunks_by_section(self, section_name):
        if section_name in self.sections_map:
            texto = self.sections_map[section_name]
            self.chunks = [texto[i:i+50000] for i in range(0, len(texto), 50000)]
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.active_section_name = section_name
            if dl_model: self.chunk_embeddings = dl_model.encode(self.chunks)
            self.seen_articles.clear(); self.temporary_blacklist.clear()
            return True
        return False

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total = len(self.chunks); SCORE_THRESHOLD = 50
        score = sum([min(v, SCORE_THRESHOLD) for v in self.mastery_tracker.values()])
        return min(int((score / (total * SCORE_THRESHOLD)) * 100), 100), len(self.failed_indices), total

    def get_strict_rules(self): return "1. NO SPOILERS. 2. DEPENDENCIA DEL TEXTO."
    def get_calibration_instructions(self): return "NO CHIVATEAR."

    def generate_case(self):
        if not self.api_key: return {"error": "Falta Llave"}
        if not self.chunks: return {"error": "Falta Norma"}
        
        idx = -1
        if self.last_failed_embedding is not None and self.chunk_embeddings is not None and not self.simulacro_mode:
            sims = cosine_similarity([self.last_failed_embedding], self.chunk_embeddings)[0]
            candidatos = [(i, s) for i, s in enumerate(sims) if self.mastery_tracker.get(i, 0) < 3]
            candidatos.sort(key=lambda x: x[1], reverse=True); 
            if candidatos: idx = candidatos[0][0]
        
        if idx == -1: idx = random.choice(range(len(self.chunks)))
        self.current_chunk_idx = idx; texto_base = self.chunks[idx]
        
        patron_articulo = r'^\s*(?:ARTÍCULO|ARTICULO|ART)\.?\s*(\d+[A-Z]?)'
        matches = list(re.finditer(patron_articulo, texto_base, re.IGNORECASE | re.MULTILINE))
        texto_final_ia = texto_base
        
        if matches:
            candidatos = [m for m in matches if m.group(0).upper().strip() not in self.seen_articles and m.group(0).upper().strip() not in self.temporary_blacklist]
            if not candidatos:
                candidatos = [m for m in matches if m.group(0).upper().strip() not in self.temporary_blacklist]
                if not candidatos: candidatos = matches; self.temporary_blacklist.clear()
                self.seen_articles.clear()
            
            sel = random.choice(candidatos)
            
            start = sel.start(); idx_m = matches.index(sel)
            end = matches[idx_m+1].start() if idx_m+1 < len(matches) else min(len(texto_base), start + 4000)
            texto_final_ia = texto_base[start:end] 
            self.current_article_label = sel.group(0).upper().strip()

            # Micro-Segmentación v71
            patron_item = r'(^\s*\d+\.\s+|^\s*[a-z]\)\s+)'
            sub_matches = list(re.finditer(patron_item, texto_final_ia, re.MULTILINE))
            if len(sub_matches) > 1:
                sel_sub = random.choice(sub_matches)
                start_sub = sel_sub.start(); idx_sub = sub_matches.index(sel_sub)
                end_sub = sub_matches[idx_sub+1].start() if idx_sub + 1 < len(sub_matches) else len(texto_final_ia)
                
                texto_fragmento = texto_final_ia[start_sub:end_sub]
                id_sub = sel_sub.group(0).strip()
                
                encabezado = texto_final_ia[:100].split('\n')[0] 
                if "ARTÍCULO" not in encabezado.upper(): encabezado = f"{self.current_article_label} (Contexto)"
                
                texto_final_ia = f"{encabezado}\n[...]\n{texto_fragmento}"
                self.current_article_label = f"{self.current_article_label} - ITEM {id_sub}"
        else:
            self.current_article_label = "General"; texto_final_ia = texto_base[:4000]

        inst_estilo = "ESTILO: TÉCNICO." if "Sin Caso" in self.structure_type else "ESTILO: NARRATIVO."
        
        # --- RECONEXIÓN DE CALIBRACIÓN (v74) ---
        feedback_instr = ""
        if self.feedback_history:
            last_feeds = self.feedback_history[-5:] # Tomamos los últimos 5 reclamos
            instrucciones_correccion = []
            if "pregunta_facil" in last_feeds: instrucciones_correccion.append("AUMENTAR DRASTICAMENTE LA DIFICULTAD.")
            if "respuesta_obvia" in last_feeds: instrucciones_correccion.append("OPCIONES TRAMPA OBLIGATORIAS. PROHIBIDO RESPUESTAS EVIDENTES.")
            if "spoiler" in last_feeds: instrucciones_correccion.append("EL ENUNCIADO NO PUEDE CONTENER PISTAS DE LA RESPUESTA.")
            if "desconexion" in last_feeds: instrucciones_correccion.append("LA PREGUNTA DEBE ESTAR 100% VINCULADA AL CASO Y TEXTO.")
            
            if instrucciones_correccion:
                feedback_instr = "CORRECCIONES DEL USUARIO (PRIORIDAD MAXIMA): " + " ".join(instrucciones_correccion)

        prompt = f"""
        ACTÚA COMO EXPERTO (NIVEL {self.level.upper()}). ENTIDAD: {self.entity.upper()}.
        {inst_estilo}
        {feedback_instr}
        Genera {self.questions_per_case} preguntas del texto.
        REGLAS:
        1. 4 OPCIONES (A,B,C,D).
        2. FUENTE EXPLÍCITA.
        3. TIP MEMORIA: Incluye 'tip_memoria'.
        4. EXPLICACIÓN ESTRUCTURADA (JSON): Explicaciones SEPARADAS por opción.
        
        EJEMPLO: '''{self.example_question}'''
        NORMA: "{texto_final_ia}"
        
        FORMATO JSON OBLIGATORIO:
        {{
            "articulo_fuente": "ARTÍCULO X",
            "narrativa_caso": "...",
            "preguntas": [
                {{ 
                    "enunciado": "...", 
                    "opciones": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, 
                    "respuesta": "A", 
                    "tip_memoria": "...",
                    "explicaciones": {{ "A": "...", "B": "...", "C": "...", "D": "..." }}
                }}
            ]
        }}
        """
        
        max_retries = 3; attempts = 0
        while attempts < max_retries:
            try:
                if self.provider == "OpenAI":
                    h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    d = {"model": "gpt-4o", "messages": [{"role": "system", "content": "JSON ONLY"}, {"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
                    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=h, json=d); text_resp = resp.json()['choices'][0]['message']['content']
                elif self.provider == "Google":
                    res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"}); text_resp = res.text.strip()
                else:
                    h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    d = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": "JSON ONLY"}, {"role": "user", "content": prompt}]}
                    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=h, json=d); text_resp = resp.json()['choices'][0]['message']['content']

                if "```" in text_resp: text_resp = re.search(r'```(?:json)?(.*?)```', text_resp, re.DOTALL).group(1).strip()
                final_json = json.loads(text_resp)
                
                if "articulo_fuente" in final_json: self.current_article_label = final_json["articulo_fuente"].upper()
                if "ITEM" in self.current_article_label and "ITEM" not in final_json.get("articulo_fuente", "").upper(): pass
                elif "articulo_fuente" in final_json: self.current_article_label = final_json["articulo_fuente"].upper()

                # Barajador
                for q in final_json['preguntas']:
                    opciones_raw = list(q['opciones'].items()) 
                    explicaciones_raw = q.get('explicaciones', {})
                    respuesta_txt = q['opciones'][q['respuesta']]
                    tip_memoria = q.get('tip_memoria', "")
                    
                    items = []
                    for k, v in opciones_raw:
                        items.append({ "txt": v, "exp": explicaciones_raw.get(k, "Sin detalle."), "ok": (v == respuesta_txt) })
                    
                    random.shuffle(items)
                    
                    nuevas_ops = {}; nueva_resp = "A"; txt_exp = ""; letras = ['A', 'B', 'C', 'D']
                    
                    for i, item in enumerate(items):
                        if i < 4:
                            letra = letras[i]; nuevas_ops[letra] = item["txt"]
                            estado = "❌ INCORRECTA"
                            if item["ok"]: nueva_resp = letra; estado = "✅ CORRECTA"
                            txt_exp += f"**({letra}) {estado}:** {item['exp']}\n\n"
                    
                    q['opciones'] = nuevas_ops; q['respuesta'] = nueva_resp; q['explicacion'] = txt_exp; q['tip_final'] = tip_memoria

                return final_json
            except Exception as e: time.sleep(1); attempts += 1
        return {"error": "Saturado."}

# ==========================================
# INTERFAZ
# ==========================================
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'case_id' not in st.session_state: st.session_state.case_id = 0
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

with st.sidebar:
    st.title("⚖️ TITÁN v74")
    with st.expander("🔑 LLAVE", expanded=True):
        key = st.text_input("API Key:", type="password")
        if key: ok, msg = engine.configure_api(key); st.success(msg) if ok else st.error(msg)
    
    st.divider()
    
    if engine.failed_articles:
        st.markdown("### 🔴 REPASAR")
        st.markdown(" ".join([f"<span class='failed-tag'>{f}</span>" for f in engine.failed_articles]), unsafe_allow_html=True)
    
    if engine.mastered_articles:
        st.markdown("### 🟢 DOMINADOS")
        st.markdown(" ".join([f"<span class='mastered-tag'>{f}</span>" for f in engine.mastered_articles]), unsafe_allow_html=True)
    
    st.divider()
    engine.study_phase = st.radio("Fase:", ["Pre-Guía", "Post-Guía"], index=0 if engine.study_phase == "Pre-Guía" else 1)
    engine.structure_type = st.radio("Estilo:", ["Técnico / Normativo", "Narrativo / Caso"], index=0 if "Sin Caso" in engine.structure_type else 1)
    engine.questions_per_case = st.number_input("Preguntas:", 1, 5, engine.questions_per_case)

    with st.expander("Detalles", expanded=True):
        if "Caso" in engine.structure_type: engine.job_functions = st.text_area("Funciones:", value=engine.job_functions, height=70)
        else: engine.example_question = st.text_area("Ejemplo:", value=engine.example_question, height=70)

    t1, t2 = st.tabs(["📝 NUEVA", "📂 CARGAR"])
    with t1:
        axis = st.text_input("Eje Temático:", value=engine.thematic_axis); txt = st.text_area("Norma:", height=100)
        if st.button("🚀 PROCESAR"): 
            if engine.process_law(txt, axis): st.session_state.page='game'; st.session_state.current_data=None; st.rerun()
    with t2:
        upl = st.file_uploader("JSON:", type=['json'])
        if upl and ('last_loaded' not in st.session_state or st.session_state.last_loaded != upl.name):
            try:
                d = json.load(upl); engine.chunks = d['chunks']; engine.sections_map = d.get('sections', {})
                engine.active_section_name = d.get('act_sec', "Todo")
                engine.failed_articles = set(d.get('failed_arts', [])); engine.mastered_articles = set(d.get('mastered_arts', []))
                engine.seen_articles = set(d.get('seen_arts', []))
                if DL_AVAILABLE: engine.chunk_embeddings = dl_model.encode(engine.chunks)
                st.session_state.last_loaded = upl.name; st.session_state.page='game'; st.session_state.current_data=None; st.rerun()
            except: st.error("Error archivo")

    if engine.chunks:
        st.download_button("💾 Guardar", json.dumps({
            "chunks": engine.chunks, "sections": engine.sections_map, 
            "failed_arts": list(engine.failed_articles), "mastered_arts": list(engine.mastered_articles),
            "seen_arts": list(engine.seen_articles)
        }), "backup.json")

if engine.sections_map and len(engine.sections_map) > 1:
        st.divider()
        st.markdown("### 📍 MAPA DE LA LEY")
        opciones = list(engine.sections_map.keys())
        if "Todo el Documento" in opciones: opciones.remove("Todo el Documento"); opciones.insert(0, "Todo el Documento")
        try: idx_sec = opciones.index(engine.active_section_name)
        except: idx_sec = 0
        seleccion = st.selectbox("Estudiar Específicamente:", opciones, index=idx_sec)
        if seleccion != engine.active_section_name:
            if engine.update_chunks_by_section(seleccion): st.session_state.current_data = None; st.rerun()

if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Dominio", f"{perc}%"); c2.metric("❌ Fallos", f"{fails}"); c3.metric("📉 Vistos", f"{len([x for x in engine.mastery_tracker.values() if x > 0])}/{total}")
    
    st.info(f"🎯 ENFOQUE: **{engine.current_article_label}**")
    st.progress(perc/100)

    if not st.session_state.get('current_data'):
        with st.spinner("🧠 Generando..."):
            data = engine.generate_case()
            if data and "preguntas" in data:
                st.session_state.case_id += 1; st.session_state.current_data = data
                st.session_state.q_idx = 0; st.session_state.answered = False; st.rerun()
            else: st.error("Error"); st.stop()

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
            with col_val: submitted = st.form_submit_button("✅ VALIDAR RESPUESTA")
            with col_skip: skipped = st.form_submit_button("⏭️ SALTAR (BLOQUEAR)")
            
            if skipped:
                engine.temporary_blacklist.add(engine.current_article_label.split(" - ITEM")[0].strip())
                st.toast(f"🚫 {engine.current_article_label} bloqueado por hoy.", icon="🗑️")
                st.session_state.current_data = None; st.rerun()

            if submitted:
                if not sel: st.warning("Selecciona una opción")
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
                        if engine.chunk_embeddings is not None: engine.last_failed_embedding = engine.chunk_embeddings[engine.current_chunk_idx]
                        if engine.current_article_label != "General":
                            if full_tag in engine.mastered_articles: engine.mastered_articles.remove(full_tag)
                            engine.failed_articles.add(full_tag)
                    
                    st.info(q['explicacion'])
                    if 'tip_final' in q and q['tip_final']: st.warning(f"💡 **TIP DE MAESTRO:** {q['tip_final']}")
                    st.session_state.answered = True

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
            errores_sel = st.multiselect("Reportar fallos:", list(reasons_map.keys()))
            if st.button("¡Castigar y Corregir!"):
                for r in errores_sel: engine.feedback_history.append(reasons_map[r])
                st.toast(f"Feedback enviado: {len(errores_sel)} error(es)", icon="🛡️")