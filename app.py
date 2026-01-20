import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- 1. CONFIGURACIÓN VISUAL ROBUSTA ---
st.set_page_config(page_title="Entrenador Legal TITÁN v6.3", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.02);}
    .narrative-box {
        background-color: #f3e5f5; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 6px solid #8e24aa; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        font-family: 'Georgia', serif;
        font-size: 1.15em;
    }
    .question-card {
        background-color: #ffffff;
        padding: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .status-bar {font-weight: bold; color: #2e86c1;}
</style>
""", unsafe_allow_html=True)

# --- 2. CEREBRO LÓGICO (TITÁN ESTABLE v6.3) ---
class LegalEngineTITAN:
    def __init__(self):
        self.chunks = []           
        self.chunk_origins = {}    
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

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            selected = next((m for m in models if 'gemini-1.5-flash' in m.lower()), None)
            if not selected: selected = next((m for m in models if 'flash' in m.lower()), models[0])
            self.model = genai.GenerativeModel(selected)
            return True, f"Conectado al Cerebro: {selected.split('/')[-1]}"
        except Exception as e:
            return False, f"Error Crítico API: {str(e)}"

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = []
        step = 5500 
        for i in range(0, len(text), step):
            chunk = text[i:i+step]
            new_chunks.append(chunk)
            
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            self.mistakes_log = []
        else:
            start_index = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)):
                real_idx = start_index + i
                self.mastery_tracker[real_idx] = 0 
        
        return len(new_chunks)

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total_chunks = len(self.chunks)
        goal_score = total_chunks * 3 
        
        # --- FIX v6.3: TOPE MATEMÁTICO ---
        # Limitamos el aporte de cada bloque a un máximo de 3 puntos.
        # Si repasaste 10 veces, solo cuentan 3 para la barra de progreso.
        current_score = sum([min(v, 3) for v in self.mastery_tracker.values()])
        
        percentage = int((current_score / goal_score) * 100) if goal_score > 0 else 0
        # Doble seguridad: Si por alguna razón sigue pasando 100, lo forzamos a 100.
        percentage = min(percentage, 100)
        
        pending_reviews = len(self.failed_indices)
        return percentage, pending_reviews, total_chunks

    def get_calibration_prompt(self):
        if not self.feedback_history: return "Modo: Estándar."
        counts = Counter(self.feedback_history)
        instructions = []
        
        instructions.append("🛡️ ANTI-RECORTE: Si el texto dice 'A y B', es PROHIBIDO generar una respuesta que diga 'Solo A'.")

        if counts['desconectado'] > 0:
            instructions.append("🔗 ANTI-SPOILER: La pregunta NO puede regalar el dato clave. El usuario debe buscarlo en la historia.")

        if counts['sesgo_longitud'] > 0:
            instructions.append("🛑 FORMATO VISUAL: Las opciones A, B y C deben tener EXACTAMENTE el mismo número de palabras (+/- 2).")

        if counts['respuesta_obvia'] > 0:
            instructions.append("💀 DIFICULTAD TÉCNICA: Los distractores deben ser trampas sutiles. Prohibido opciones absurdas.")
        
        if counts['pregunta_facil'] > 0:
            instructions.append("⚠️ TRAMPA DE DETALLE: La respuesta correcta debe depender de un dato pequeño escondido en el texto.")
            
        if counts['repetitivo'] > 0:
            self.current_temperature = 0.7 
            instructions.append("🔄 VARIEDAD TOTAL: Cambia nombres, cargos y la situación problema.")
        
        if counts['alucinacion'] > 0:
            self.current_temperature = 0.0 
            instructions.append("⛔ FUENTE CERRADA: Usa SOLO lo que está escrito en el texto.")

        return "\n".join(instructions)

    def _safe_call(self, prompt):
        retries = 3
        wait = 5
        config = {"temperature": self.current_temperature, "response_mime_type": "application/json"}
        
        for i in range(retries):
            try:
                response = self.model.generate_content(prompt, generation_config=config)
                return response.text
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    st.toast(f"⏳ Tráfico alto IA. Reintentando en {wait}s...", icon="🚦")
                    time.sleep(wait)
                    wait *= 2
                else:
                    return None
        return None

    def generate_case(self):
        if not self.chunks: return {"error": "⚠️ Carga una norma primero."}
        
        if self.simulacro_mode:
            idx = random.choice(range(len(self.chunks)))
        else:
            if self.failed_indices:
                if random.random() < 0.6: idx = random.choice(list(self.failed_indices))
                else:
                    pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                    idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        text_chunk = self.chunks[idx]
        current_level = self.mastery_tracker.get(idx, 0)
        
        lentes = ["NIVEL 1: CONCEPTUAL", "NIVEL 2: PROCEDIMENTAL", "NIVEL 3: SANCIONATORIO", "NIVEL 4: SITUACIONAL"]
        lente_actual = lentes[min(current_level, 3)]
        contexto = f"CONTEXTO: {self.entity.upper()}" if self.entity else ""
        
        instruction_level = ""
        if self.level == "Asistencial":
            instruction_level = "NIVEL BÁSICO/EJECUCIÓN: Preguntas sobre pasos y definiciones literales."
        elif self.level == "Técnico":
            instruction_level = "NIVEL TÉCNICO: Preguntas sobre aplicación de procesos."
        elif self.level == "Profesional":
            instruction_level = "NIVEL PROFESIONAL: Preguntas de ANÁLISIS y CRITERIO. Prohibido definiciones. Evalúa interpretación compleja."
        elif self.level == "Asesor":
            instruction_level = "NIVEL ASESOR: Preguntas de ESTRATEGIA y RIESGO. El caso debe tener vacíos que resolver con principios."

        calibracion_activa = self.get_calibration_prompt()

        prompt = f"""
        ACTÚA COMO UN EXPERTO DISEÑADOR DE PRUEBAS CNSC PARA EL NIVEL: **{self.level.upper()}**.
        
        TEXTO FUENTE:
        ---------------------------------------------------------
        "{text_chunk[:6000]}"
        ---------------------------------------------------------
        
        MISIÓN: Crear un CASO SITUACIONAL con 4 PREGUNTAS ajustadas al nivel **{self.level}**.
        
        INSTRUCCIÓN DE NIVEL ({self.level.upper()}):
        {instruction_level}
        
        REGLAS DE ORO:
        1. **INTEGRIDAD:** No recortes requisitos.
        2. **ANTI-SPOILER:** No regales el dato clave en la pregunta.
        3. **VINCULACIÓN:** "Analizando la conducta de [PERSONAJE] en la fecha [FECHA]...".
        
        !!! AJUSTES DEL USUARIO !!!
        {calibracion_activa}
        
        FORMATO JSON OBLIGATORIO:
        {{
            "narrativa_caso": "Narración detallada con fechas, nombres y situaciones ocultas...",
            "preguntas": [
                {{
                    "enunciado": "Pregunta de nivel {self.level}...",
                    "opciones": {{ "A": "...", "B": "...", "C": "..." }},
                    "respuesta": "A",
                    "explicacion": "..."
                }},
                ... (Total 4 preguntas)
            ]
        }}
        """
        
        res_json = self._safe_call(prompt)
        if not res_json: return {"error": "Error de conexión."}
        
        try:
            clean_text = res_json.strip()
            if "```" in clean_text:
                clean_text = re.search(r'```(?:json)?(.*?)```', clean_text, re.DOTALL)
                if clean_text: clean_text = clean_text.group(1).strip()
            return json.loads(clean_text)
        except:
            return {"error": "Error procesando respuesta IA."}

# --- 3. INICIALIZACIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False

engine = st.session_state.engine

# --- 4. INTERFAZ ---
with st.sidebar:
    st.title("⚙️ Panel de Control")
    key = ""
    if "GEMINI_KEY" in st.secrets:
        key = st.secrets["GEMINI_KEY"]
        st.success("🔑 Licencia Activa")
    else:
        key = st.text_input("Ingresa tu API Key:", type="password")
    
    if key and not engine.model:
        ok, msg = engine.configure_api(key)
        if not ok: st.error(msg)
    
    st.divider()
    
    st.markdown("### 🎯 Nivel del Cargo")
    niveles = ["Asistencial", "Técnico", "Profesional", "Asesor"]
    engine.level = st.selectbox("Selecciona tu Nivel:", niveles, index=2)
    st.info(f"Modo activado: **{engine.level}**")
    
    st.divider()
    engine.entity = st.text_input("Entidad:", placeholder="Ej: Fiscalía...")
    txt_input = st.text_area("Texto de la Norma:", height=200)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚠️ INICIAR NUEVO", type="primary"):
            c = engine.process_law(txt_input, append=False)
            if c: 
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.rerun()
            
    with col2:
        if st.button("➕ AGREGAR"):
            c = engine.process_law(txt_input, append=True)
            if c: st.success(f"+{c} Bloques.")
            
    st.divider()
    
    if st.button("🗑️ Borrar Historial Calibración"):
        engine.feedback_history = []
        st.toast("Memoria limpia.", icon="🧼")
    
    st.divider()
    if st.button("🔥 MODO SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True
        st.session_state.current_data = None
        st.session_state.page = 'game'
        st.rerun()
        
    st.divider()
    if engine.chunks:
        save_data = json.dumps({"chunks": engine.chunks, "mastery": engine.mastery_tracker, "failed": list(engine.failed_indices), "log": engine.mistakes_log, "feed": engine.feedback_history, "entity": engine.entity})
        st.download_button("Descargar JSON", save_data, "progreso.json", "application/json")
    
    upl = st.file_uploader("Cargar JSON", type=['json'])
    if upl:
        try:
            d = json.load(upl)
            engine.chunks = d['chunks']
            engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
            engine.failed_indices = set(d['failed'])
            engine.mistakes_log = d['log']
            engine.feedback_history = d['feed']
            engine.entity = d.get('entity', "")
            st.success("¡Recuperado!")
        except: st.error("Error.")

# --- 5. JUEGO ---
if st.session_state.page == 'game':
    perc, fails, total = engine.get_stats()
    
    # --- BARRA DE PROGRESO SEGURA ---
    # Convertimos a float entre 0.0 y 1.0 para st.progress
    safe_perc = min(float(perc) / 100.0, 1.0)
    
    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; background:#eee; padding:10px; border-radius:8px;'>
        <span class='status-bar'>{'🔥 SIMULACRO' if engine.simulacro_mode else '📚 ESTUDIO'}</span>
        <span class='status-bar'>NIVEL: {engine.level.upper()}</span>
        <span class='status-bar'>DOMINIO: {perc}%</span>
        <span class='status-bar'>BLOQUES: {total}</span>
        <span class='status-bar' style='color:red'>REPASOS: {fails}</span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        st.progress(safe_perc)
    except:
        st.progress(1.0) # Fallback por si acaso

    if not st.session_state.current_data:
        with st.spinner(f"⚖️ Diseñando caso nivel {engine.level.upper()}..."):
            data = engine.generate_case()
            if "error" in data:
                st.error(data['error'])
                if st.button("Reintentar"): st.rerun()
                st.stop()
            st.session_state.current_data = data
            st.session_state.q_idx = 0
            st.session_state.answered = False
            st.rerun()

    data = st.session_state.current_data
    q_idx = st.session_state.q_idx
    questions = data.get('preguntas', [])

    st.markdown(f"""
    <div class="narrative-box">
        <h4>📜 Caso Situacional ({engine.level})</h4>
        <p style="font-size:1.1em; line-height:1.6;">{data.get('narrativa_caso', 'Error narrativo.')}</p>
    </div>
    """, unsafe_allow_html=True)

    if q_idx < len(questions):
        q = questions[q_idx]
        with st.container():
            st.markdown(f"### 🔹 Pregunta {q_idx + 1} de {len(questions)}")
            st.markdown(f"##### {q['enunciado']}")
            opts = q['opciones']
            op_list = [f"A) {opts.get('A','')}", f"B) {opts.get('B','')}", f"C) {opts.get('C','')}"]
            
            with st.form(key=f"q_form_{q_idx}"):
                selection = st.radio("Respuesta:", op_list, index=None)
                if st.form_submit_button("✅ Validar Respuesta") and selection:
                    letter = selection[0]
                    correct = q['respuesta'].upper()
                    if letter == correct:
                        st.success("✅ ¡CORRECTO!")
                        if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                    else:
                        st.error(f"❌ INCORRECTO. Era la {correct}.")
                        engine.failed_indices.add(engine.current_chunk_idx)
                        engine.mistakes_log.append({"pregunta": q['enunciado'], "error": letter, "correcta": correct})
                    st.info(f"💡 {q['explicacion']}")
                    st.session_state.answered = True

        if st.session_state.answered:
            col_nav, col_rep = st.columns([1, 1])
            with col_nav:
                if q_idx < len(questions) - 1:
                    if st.button("⏭️ Siguiente Pregunta"):
                        st.session_state.q_idx += 1
                        st.session_state.answered = False
                        st.rerun()
                else:
                    if st.button("🔄 TERMINAR CASO"):
                        if not engine.simulacro_mode:
                            idx = engine.current_chunk_idx
                            engine.mastery_tracker[idx] = engine.mastery_tracker.get(idx, 0) + 1
                        st.session_state.current_data = None
                        st.session_state.q_idx = 0
                        st.session_state.answered = False
                        st.rerun()
            
            with col_rep:
                with st.expander("📢 Calibrar IA (REPORTAR FALLO)"):
                    reasons = {
                        "Respuesta Incompleta (Recortó la norma)": "alucinacion",
                        "Pregunta con Spoiler (Regala el dato)": "desconectado",
                        "Respuesta muy Obvia": "respuesta_obvia",
                        "Opciones de diferente largo": "sesgo_longitud",
                        "Pregunta muy Fácil": "pregunta_facil",
                        "Repetitivo": "repetitivo"
                    }
                    selected_reason = st.selectbox("¿Qué falló?", list(reasons.keys()))
                    
                    if st.button("Enviar y Ajustar"):
                        code = reasons[selected_reason]
                        engine.feedback_history.append(code)
                        if code == "alucinacion": st.toast("Filtro de Integridad Activado.", icon="🛡️")
                        elif code == "desconectado": st.toast("Candado Anti-Spoiler Ajustado.", icon="🤐")
                        else: st.toast("Ajuste Acumulado.", icon="✅")

elif st.session_state.page == 'setup':
    st.markdown("<h1>🏛️ Entrenador Legal TITÁN v6.3</h1>", unsafe_allow_html=True)