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

# --- CLASE MOTOR (CEREBRO v4.0 - FUENTE CERRADA ESTRICTA) ---
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
        # Temperatura muy baja por defecto para evitar alucinaciones
        self.current_temperature = 0.1 

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
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
        # Bloques grandes (5000 chars) para asegurar contexto completo
        step = 5000 
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
        if not self.feedback_history: return "Modo: Estricto."
        counts = Counter(self.feedback_history)
        instructions = []
        
        # --- LÓGICA DE CALIBRACIÓN v4.0 ---
        if counts['alucinacion'] > 0:
            self.current_temperature = 0.0 # CERO creatividad (Robot estricto)
            instructions.append("⛔ CANDADO ACTIVADO: El usuario reportó invención. REGLA SUPREMA: Si la respuesta no está LITERALMENTE en el texto, no hagas la pregunta.")
        
        if counts['repetitivo'] > 0:
            self.current_temperature = 0.5
            instructions.append("⚠️ Usa el mismo texto normativo pero cambia la situación fáctica (nombres/lugares) para dar variedad.")
            
        if counts['pregunta_facil'] > 0: 
            instructions.append("⚠️ DIFICULTAD: Las opciones falsas deben ser jurídicamente viables pero incorrectas por un detalle sutil del texto.")

        return "\n".join(instructions)

    def _safe_generate(self, prompt, is_json=False):
        max_retries = 3
        wait_time = 5
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
                if "429" in error_str or "quota" in error_str:
                    with st.spinner(f"⏳ Esperando a Google ({wait_time}s)..."):
                        time.sleep(wait_time)
                    wait_time *= 2
                else:
                    return None
        return None

    def generate_adaptive_case(self):
        if not self.chunks: return {"error": "⚠️ No hay ley cargada."}
        
        # SELECCIÓN
        if self.simulacro_mode:
            current_idx = random.choice(range(len(self.chunks)))
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
        
        contexto_entidad = ""
        if self.entity.strip():
            contexto_entidad = f"CONTEXTO: {self.entity.upper()} (Úsalo solo para dar ambiente, NO para inventar normas)."

        # --- PROMPT v4.0: NOTARIO ESTRICTO ---
        prompt = f"""
        ACTÚA COMO UN EXAMINADOR ESTRICTO TIPO 'FUENTE CERRADA'.
        
        TU ÚNICA FUENTE DE VERDAD ES EL SIGUIENTE TEXTO.
        OLVIDA CUALQUIER OTRA LEY O CONOCIMIENTO EXTERNO QUE TENGAS.
        SI ALGO NO ESTÁ EN EL TEXTO, NO EXISTE.

        === TEXTO SAGRADO (LEER DETENIDAMENTE) ===
        "{active_text}"
        ==========================================

        OBJETIVO: Crear un caso situacional basado 100% en el texto anterior.

        INSTRUCCIONES DE DISEÑO:
        1. **FOCO:** {lente_actual}.
        2. **AMBIENTACIÓN:** {contexto_entidad}
        3. **VALIDACIÓN OBLIGATORIA:** Para cada pregunta, debes ser capaz de subrayar la frase del texto que da la respuesta. Si no puedes subrayarla, borra la pregunta.
        4. **ANTI-ALUCINACIÓN:** No preguntes sobre temas generales (Constitución, CPACA) a menos que el "TEXTO SAGRADO" los mencione explícitamente.
        5. **ESTILO:** El caso debe aplicar la norma del texto a una situación con personajes ficticios (Juan, María, el Director, etc.).

        FEEDBACK DE CALIBRACIÓN:
        {self.get_learning_context()}

        FORMATO JSON OBLIGATORIO:
        {{
            "narrativa_caso": "Historia que aplica el texto...",
            "preguntas": [
                {{ 
                    "enunciado": "Pregunta...", 
                    "opciones": {{ "A": "..", "B": "..", "C": ".." }}, 
                    "respuesta": "A", 
                    "explicacion": "CORRECTA PORQUE EL TEXTO DICE LITERALMENTE: '...[citar fragmento del texto aquí]...'" 
                }}
            ]
        }}
        """
        json_res = self._safe_generate(prompt, is_json=True)
        if not json_res: return {"error": "Error de conexión."}
        try:
            text = json_res.strip()
            if "```" in text:
                text = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
                if text: text = text.group(1).strip()
            return json.loads(text)
        except: return {"error": "Error procesando respuesta."}

# --- INICIALIZACIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEnginePRO()
if 'page' not in st.session_state: st.session_state.page = 'setup'
engine = st.session_state.engine

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    api_key_input = ""
    if "GEMINI_KEY" in st.secrets:
        api_key_input = st.secrets["GEMINI_KEY"]
        st.success("🔑 Key Auto-cargada")
    else:
        api_key_input = st.text_input("Gemini API Key", type="password")
    
    if api_key_input and not engine.model:
        ok, msg = engine.configure_api(api_key_input)
        if not ok: st.error(msg)
    
    st.divider()
    engine.entity = st.text_input("Entidad", placeholder="Ej: DIAN, Contraloría...")
    uploaded_text = st.text_area("Pegar Norma", height=150)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚠️ Reiniciar", type="primary"):
            if engine.process_law(uploaded_text, append=False) > 0:
                engine.simulacro_mode = False
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.rerun()
    with c2:
        if st.button("➕ Agregar"):
            c = engine.process_law(uploaded_text, append=True)
            if c > 0: st.success(f"+{c}")

    st.divider()
    if st.button("🔥 MODO SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True
        st.session_state.current_data = None
        st.session_state.page = 'game'
        st.rerun()

    st.divider()
    if engine.chunks:
        data_to_save = {"chunks": engine.chunks, "mastery": {str(k):v for k,v in engine.mastery_tracker.items()}, "failed": list(engine.failed_indices), "log": engine.mistakes_log, "feed": engine.feedback_history, "entity": engine.entity}
        st.download_button("💾 Bajar", json.dumps(data_to_save), "backup.json", "application/json")
    
    uf = st.file_uploader("📂 Subir", type=['json'])
    if uf:
        try:
            d = json.load(uf)
            engine.chunks = d['chunks']
            engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
            engine.failed_indices = set(d['failed'])
            engine.mistakes_log = d['log']
            engine.feedback_history = d['feed']
            engine.entity = d.get('entity', "")
            st.success("Cargado")
        except: st.error("Error")

# --- PÁGINA JUEGO ---
if st.session_state.page == 'game':
    perc, fails = engine.get_progress_stats()
    st.progress(perc/100, f"Dominio: {perc}% | Pendientes: {fails} | Modo: {'SIMULACRO' if engine.simulacro_mode else 'ESTUDIO'}")

    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        with st.spinner("⚖️ Analizando texto estricto..."):
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

    with st.container(border=True):
        st.markdown(f"#### 📜 Caso: {data.get('narrativa_caso', 'Error')}")
    
    if q_idx < len(data.get('preguntas', [])):
        q = data['preguntas'][q_idx]
        st.subheader(f"Pregunta {q_idx + 1}")
        st.markdown(f"**{q['enunciado']}**")
        opts = q['opciones']
        opt_list = [f"A) {opts.get('A','')}", f"B) {opts.get('B','')}", f"C) {opts.get('C','')}"]
        
        with st.form("game"):
            choice = st.radio("Respuesta:", opt_list, index=None)
            if st.form_submit_button("✅ Validar") and choice:
                sel = choice[0]
                corr = q['respuesta'].upper()
                if sel == corr:
                    st.success("✅ ¡Correcto!")
                    if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                else:
                    st.error(f"❌ Incorrecto. Era {corr}.")
                    engine.failed_indices.add(engine.current_chunk_idx)
                    engine.mistakes_log.append({"pregunta": q['enunciado'], "elegida": sel, "correcta": corr})
                st.info(f"💡 {q['explicacion']}")
                st.session_state.answered = True

        if st.session_state.answered:
            c1, c2 = st.columns([1,1])
            with c1:
                lbl = "⏭️ Siguiente" if q_idx < len(data['preguntas'])-1 else "🔄 Nuevo Caso"
                if st.button(lbl):
                    if q_idx < len(data['preguntas'])-1: st.session_state.q_idx += 1
                    else:
                        if not engine.simulacro_mode and engine.current_chunk_idx not in engine.failed_indices:
                            engine.mastery_tracker[engine.current_chunk_idx] = engine.mastery_tracker.get(engine.current_chunk_idx, 0) + 1
                        st.session_state.current_data = None
                    st.session_state.answered = False
                    st.rerun()
            with c2:
                with st.expander("📢 Reportar Error"):
                    reason = st.selectbox("Error:", ["alucinacion", "repetitivo", "pregunta_facil", "caso_simple", "error_estructural"])
                    if st.button("Enviar"):
                        engine.feedback_history.append(reason)
                        st.toast("Calibrando restricciones...", icon="🔒")

elif st.session_state.page == 'setup':
    st.title("🏛️ Entrenador Legal PRO v4.0")
    st.info("👈 Pega tu norma. El sistema ignorará cualquier conocimiento externo.")