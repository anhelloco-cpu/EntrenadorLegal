import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Entrenador Legal PRO", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    .success-box {padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; margin-bottom: 10px;}
    .report-box {border: 2px solid #ff4b4b; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- INYECCIÓN DE VARIEDAD (LISTAS DE PYTHON) ---
# Estas listas fuerzan a la IA a salir de su zona de confort
ROLES = [
    "un Celador", "la Secretaria General", "el Alcalde", "un Contratista de obra", 
    "un Veedor Ciudadano", "un Juez de Paz", "un Policía de tránsito", "un Inspector de Obra", 
    "la Tesorera", "un Concejal opositor", "el Almacenista", "un Conductor de la entidad"
]
SITUACIONES = [
    "una licitación declarada desierta", "la pérdida misteriosa de un disco duro", 
    "un regalo navideño costoso dejado en el escritorio", "una insinuación de acoso laboral", 
    "un vencimiento de términos ocurrido ayer", "una firma que parece falsificada", 
    "una orden verbal ilegal del superior", "una incapacidad médica presuntamente falsa",
    "un hallazgo fiscal de la Contraloría", "una tutela por derecho de petición"
]

# --- CLASE MOTOR (CEREBRO v3.8) ---
class LegalEnginePRO:
    def __init__(self):
        self.chunks = []
        self.mastery_tracker = {}
        self.GOAL_REPETITIONS = 3
        self.failed_indices = set()
        self.mistakes_log = []
        self.feedback_history = []
        self.current_data = None
        self.current_chunk_idx = -1
        self.q_index = 0
        self.entity = ""
        self.simulacro_mode = False
        self.model = None
        # NUEVO: Temperatura dinámica para romper la repetición
        self.current_temperature = 0.3 

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Preferir modelo Flash por velocidad y cuota
            selected = next((m for m in models if 'gemini-1.5-flash' in m.lower()), None)
            if not selected: selected = next((m for m in models if 'flash' in m.lower()), models[0])
            self.model = genai.GenerativeModel(selected)
            return True, f"Conectado a: {selected.split('/')[-1]}"
        except Exception as e:
            return False, f"Error API: {str(e)}"

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        new_chunks = []
        step = 3500 
        for i in range(0, len(text), step):
            new_chunks.append(text[i:i+step])
            
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            self.mistakes_log = []
        else:
            start_index = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)):
                self.mastery_tracker[start_index + i] = 0
        return len(new_chunks)

    def get_progress_stats(self):
        if not self.chunks: return 0, 0
        total_goal = len(self.chunks) * self.GOAL_REPETITIONS
        current_progress = sum(self.mastery_tracker.values())
        fails = len(self.failed_indices)
        if total_goal == 0: return 0, 0
        percent = int((current_progress / total_goal) * 100)
        return min(percent, 100), fails

    def get_learning_context(self):
        if not self.feedback_history: return "Nivel Estándar."
        counts = Counter(self.feedback_history)
        instructions = []
        
        # --- LÓGICA DE CALIBRACIÓN AGRESIVA ---
        if counts['repetitivo'] > 0:
            self.current_temperature = 0.9 # MÁXIMA CREATIVIDAD
            instructions.append("⚠️ CRÍTICO - ANTI-REPETICIÓN: El usuario reportó casos repetidos. GENERA UN ESCENARIO BIZARRO, ATÍPICO Y COMPLEJO. No uses los nombres ni situaciones de siempre.")
        else:
            self.current_temperature = 0.4 # Temperatura normal
            
        if counts['pregunta_facil'] > 0: 
            instructions.append("⚠️ MODO CASCARITA: Las opciones incorrectas deben ser SEMÁNTICAMENTE CASI IDÉNTICAS a la correcta. Cambia solo una palabra clave (ej: 'Dolo' por 'Culpa Grave').")
        
        if counts['caso_simple'] > 0: 
            instructions.append("⚠️ COMPLEJIDAD: Involucra mínimo 3 actores y 2 fechas contradictorias en el relato.")
        
        if counts['alucinacion'] > 0: instructions.append("⚠️ CÁRCEL DE FUENTE: Cíñete 100% al texto. Si no está escrito, no existe.")
        if counts['error_estructural'] > 0: instructions.append("⚠️ ESTRUCTURA: Respeta formato JSON. Solo claves A, B, C.")
        if counts['sesgo_longitud'] > 0: instructions.append("⚠️ ANTI-SESGO: Todas las opciones deben tener EXACTAMENTE la misma longitud visual.")
        
        return "\n".join(instructions)

    def _safe_generate(self, prompt, is_json=False):
        max_retries = 3
        wait_time = 5
        # Configuración dinámica basada en el feedback
        gen_config = {
            "temperature": self.current_temperature,
            "response_mime_type": "application/json" if is_json else "text/plain"
        }
        
        for attempt in range(max_retries):
            try:
                res = self.model.generate_content(prompt, generation_config=gen_config)
                return res.text
            except Exception as e:
                error_str = str(e).lower()
                # Manejo de error 429 (Quota Exceeded)
                if "429" in error_str or "quota" in error_str:
                    with st.spinner(f"⏳ Google saturado. Esperando {wait_time}s..."):
                        time.sleep(wait_time)
                    wait_time *= 2
                else:
                    return None
        return None

    def generate_adaptive_case(self):
        if not self.chunks: return {"error": "⚠️ No hay ley cargada."}
        
        # 1. SELECCIÓN DE TEXTO
        if self.simulacro_mode:
            current_idx = random.choice(range(len(self.chunks)))
            prompt_modifier = "MODO SIMULACRO: Caso de ALTA DIFICULTAD integrando múltiples conceptos."
        else:
            if self.failed_indices and random.random() < 0.4:
                current_idx = random.choice(list(self.failed_indices))
            else:
                pending = [idx for idx, count in self.mastery_tracker.items() if count < self.GOAL_REPETITIONS]
                if not pending: current_idx = random.choice(range(len(self.chunks)))
                else: current_idx = random.choice(pending)
        
        self.current_chunk_idx = current_idx
        active_text = self.chunks[current_idx]
        current_mastery = self.mastery_tracker.get(current_idx, 0)
        
        lentes = ["CONCEPTUAL", "PROCEDIMENTAL", "SANCIONATORIO", "SITUACIONAL"]
        lente_actual = lentes[current_mastery % len(lentes)]
        
        # 2. INYECCIÓN DE ALEATORIEDAD (PYTHON)
        rol_random = random.choice(ROLES)
        sit_random = random.choice(SITUACIONES)
        
        contexto_entidad = ""
        if self.entity.strip():
            contexto_entidad = f"Ambienta el caso OBLIGATORIAMENTE en: **{self.entity.upper()}**."

        # 3. PROMPT MAESTRO
        prompt = f"""
        Actúa como un experto redactor de la Comisión Nacional del Servicio Civil (CNSC).
        Diseña una prueba de juicio situacional NIVEL DIFÍCIL.

        TEXTO NORMATIVO FUENTE: "{active_text[:4500]}"...

        === TU MISIÓN DE DISEÑO ===
        1. **PROTAGONISTA:** El caso debe tratar sobre {rol_random}.
        2. **SITUACIÓN:** El conflicto gira en torno a {sit_random}.
        3. **ENTIDAD:** {contexto_entidad}
        4. **FOCO JURÍDICO:** {lente_actual}.

        === REGLAS DE DIFICULTAD (CASCARITAS) ===
        * **NO** hagas preguntas obvias.
        * Las opciones incorrectas (distractores) deben ser JURÍDICAMENTE PLAUSIBLES.
        * Diferencia las opciones por detalles sutiles: un plazo (3 días vs 5 días), una autoridad, o la intención (Dolo vs Culpa).
        * Todas las opciones deben tener la misma longitud.

        === CALIBRACIÓN DE USUARIO (FEEDBACK) ===
        {self.get_learning_context()}

        Responde SOLO el JSON:
        {{
            "narrativa_caso": "Historia compleja con nombres y cargos específicos...",
            "preguntas": [
                {{ "enunciado": "...", "opciones": {{ "A": "..", "B": "..", "C": ".." }}, "respuesta": "A", "explicacion": ".." }},
                {{ "enunciado": "...", "opciones": {{ "A": "..", "B": "..", "C": ".." }}, "respuesta": "B", "explicacion": ".." }}
            ]
        }}
        """
        json_res = self._safe_generate(prompt, is_json=True)
        if not json_res: return {"error": "Error de conexión con Google (Intenta de nuevo)."}
        try:
            text = json_res.strip()
            if "```" in text:
                text = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
                if text: text = text.group(1).strip()
            return json.loads(text)
        except: return {"error": "Error procesando respuesta."}

# --- INICIALIZACIÓN DE SESIÓN ---
if 'engine' not in st.session_state:
    st.session_state.engine = LegalEnginePRO()
if 'page' not in st.session_state:
    st.session_state.page = 'setup'

engine = st.session_state.engine

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # 1. API KEY (SECRETS O MANUAL)
    api_key_input = ""
    if "GEMINI_KEY" in st.secrets:
        api_key_input = st.secrets["GEMINI_KEY"]
        st.success("🔑 API Key detectada (Secretos)")
    else:
        api_key_input = st.text_input("Gemini API Key", type="password")
    
    if api_key_input and not engine.model:
        ok, msg = engine.configure_api(api_key_input)
        if not ok: st.error(msg)
    
    st.divider()
    
    # 2. CARGA DE NORMA
    engine.entity = st.text_input("2. Entidad (Opcional)", placeholder="Ej: DIAN, Contraloría...")
    uploaded_text = st.text_area("3. Pegar Norma/Ley", height=150)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚠️ Reiniciar Todo", type="primary"):
            if engine.process_law(uploaded_text, append=False) > 0:
                engine.simulacro_mode = False
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.rerun()
            else: st.error("Texto muy corto")
    with c2:
        if st.button("➕ Agregar Norma"):
            c = engine.process_law(uploaded_text, append=True)
            if c > 0: st.success(f"+{c} bloques")
            else: st.error("Texto vacío")

    st.divider()
    
    # 3. MODOS
    if st.button("🔥 MODO SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True
        st.session_state.current_data = None
        st.session_state.page = 'game'
        st.toast("Modo Simulacro Activado", icon="🔥")
        st.rerun()

    st.divider()
    
    # 4. PERSISTENCIA
    st.markdown("### 💾 Guardar/Cargar")
    if engine.chunks:
        data_to_save = {
            "chunks": engine.chunks,
            "mastery": {str(k):v for k,v in engine.mastery_tracker.items()},
            "failed": list(engine.failed_indices),
            "log": engine.mistakes_log,
            "feed": engine.feedback_history,
            "entity": engine.entity
        }
        st.download_button("Descargar Progreso", json.dumps(data_to_save), "backup.json", "application/json")
    
    uf = st.file_uploader("Subir Archivo", type=['json'])
    if uf:
        try:
            d = json.load(uf)
            engine.chunks = d['chunks']
            engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
            engine.failed_indices = set(d['failed'])
            engine.mistakes_log = d['log']
            engine.feedback_history = d['feed']
            engine.entity = d.get('entity', "")
            st.success("Progreso Restaurado")
        except: st.error("Archivo inválido")

# --- PÁGINA DE JUEGO ---
if st.session_state.page == 'game':
    perc, fails = engine.get_progress_stats()
    st.progress(perc/100, f"Dominio: {perc}% | Repasos: {fails} | Modo: {'SIMULACRO' if engine.simulacro_mode else 'ESTUDIO'}")

    # Generar caso si no existe
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        with st.spinner("⚖️ Redactando caso complejo..."):
            data = engine.generate_adaptive_case()
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

    # Mostrar Caso
    with st.container(border=True):
        st.markdown(f"#### 📜 Narrativa del Caso")
        st.write(data.get('narrativa_caso', 'Error de generación.'))
    
    if q_idx < len(data.get('preguntas', [])):
        q = data['preguntas'][q_idx]
        st.subheader(f"Pregunta {q_idx + 1}")
        st.markdown(f"**{q['enunciado']}**")
        
        opts = q['opciones']
        opt_list = [f"A) {opts.get('A','')}", f"B) {opts.get('B','')}", f"C) {opts.get('C','')}"]
        
        with st.form("game_form"):
            choice = st.radio("Selecciona:", opt_list, index=None)
            submitted = st.form_submit_button("✅ Validar Respuesta")
            
            if submitted and choice:
                sel = choice[0]
                corr = q['respuesta'].upper()
                if sel == corr:
                    st.success("✅ ¡Correcto!")
                    if engine.current_chunk_idx in engine.failed_indices:
                        engine.failed_indices.remove(engine.current_chunk_idx)
                else:
                    st.error(f"❌ Incorrecto. La respuesta era {corr}.")
                    engine.failed_indices.add(engine.current_chunk_idx)
                    engine.mistakes_log.append({"pregunta": q['enunciado'], "elegida": sel, "correcta": corr})
                
                st.info(f"💡 Explicación: {q['explicacion']}")
                st.session_state.answered = True

        # Botones de Navegación y Calibración
        if st.session_state.answered:
            col1, col2 = st.columns([1, 1])
            with col1:
                label = "⏭️ Siguiente Pregunta" if q_idx < len(data['preguntas'])-1 else "🔄 Nuevo Caso"
                if st.button(label):
                    if q_idx < len(data['preguntas'])-1:
                        st.session_state.q_idx += 1
                    else:
                        if not engine.simulacro_mode and engine.current_chunk_idx not in engine.failed_indices:
                            engine.mastery_tracker[engine.current_chunk_idx] = engine.mastery_tracker.get(engine.current_chunk_idx, 0) + 1
                        st.session_state.current_data = None
                    st.session_state.answered = False
                    st.rerun()
            
            with col2:
                with st.expander("📢 Reportar Error / Ajustar Dificultad"):
                    reason = st.selectbox("Motivo del reporte:", [
                        "repetitivo", "pregunta_facil", "caso_simple", 
                        "alucinacion", "error_estructural", "sesgo_longitud", "desconectado"
                    ])
                    if st.button("Enviar Feedback"):
                        engine.feedback_history.append(reason)
                        st.toast("Feedback recibido. Ajustando algoritmo...", icon="🧠")

elif st.session_state.page == 'setup':
    st.title("🏛️ Entrenador Legal PRO v3.8")
    st.info("👈 Configura la API y carga la norma en la barra lateral.")