import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter
from io import StringIO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Entrenador Legal PRO", page_icon="⚖️", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 5px;}
    .success-box {padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; margin-bottom: 10px;}
    .error-box {padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# --- CLASE DEL MOTOR (CEREBRO v3.6) ---
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
        if not self.feedback_history: return "Nivel inicial."
        counts = Counter(self.feedback_history)
        instructions = []
        if counts['repetitivo'] > 0: instructions.append("⚠️ VARIEDAD: Cambia nombres y situaciones drásticamente.")
        if counts['caso_simple'] > 0: instructions.append("⚠️ Aumenta complejidad de HECHOS.")
        if counts['pregunta_facil'] > 0: instructions.append("⚠️ Genera distractores difíciles.")
        if counts['desconectado'] > 0: instructions.append("⚠️ Respuesta basada estrictamente en hechos.")
        if counts['alucinacion'] > 0: instructions.append("⚠️ ALERTA: NO inventes normas.")
        if counts['error_estructural'] > 0: instructions.append("⚠️ ESTRUCTURA: Solo respuestas A, B, C.")
        if counts['sesgo_longitud'] > 0: instructions.append("⚠️ ANTI-SESGO: Iguala longitud de opciones.")
        return "\n".join(instructions)

    def _safe_generate(self, prompt, is_json=False):
        max_retries = 3
        wait_time = 5
        for attempt in range(max_retries):
            try:
                config = {"response_mime_type": "application/json"} if is_json else {}
                res = self.model.generate_content(prompt, generation_config=config)
                return res.text
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str:
                    with st.spinner(f"⏳ Tráfico alto en Google. Reintentando en {wait_time}s..."):
                        time.sleep(wait_time)
                    wait_time *= 2
                else:
                    return None
        return None

    def generate_adaptive_case(self):
        if not self.chunks: return {"error": "⚠️ No hay ley cargada."}
        
        if self.simulacro_mode:
            current_idx = random.choice(range(len(self.chunks)))
            prompt_modifier = "MODO SIMULACRO: Genera un caso difícil integrando conceptos."
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
            contexto_entidad = f"Ambienta el caso en: **{self.entity.upper()}**."

        prompt = f"""
        Actúa como la Comisión Nacional del Servicio Civil (CNSC).
        TEXTO NORMATIVO BASE: "{active_text[:4500]}"...

        {contexto_entidad}
        {prompt_modifier if self.simulacro_mode else ""}

        === INSTRUCCIONES ===
        1. **FOCO:** {lente_actual}.
        2. **FUENTE CERRADA:** Usa SOLO el texto normativo base.
        3. **FORMATO:** Genera JSON con 2 preguntas. Claves A, B, C.
        
        === FEEDBACK PREVIO ===
        {self.get_learning_context()}

        Responde SOLO el JSON:
        {{
            "narrativa_caso": "Historia...",
            "preguntas": [
                {{ "enunciado": "...", "opciones": {{ "A": "..", "B": "..", "C": ".." }}, "respuesta": "A", "explicacion": ".." }},
                {{ "enunciado": "...", "opciones": {{ "A": "..", "B": "..", "C": ".." }}, "respuesta": "B", "explicacion": ".." }}
            ]
        }}
        """
        json_res = self._safe_generate(prompt, is_json=True)
        if not json_res: return {"error": "Error de conexión con Google."}
        
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
    st.session_state.page = 'setup' # setup, game, report

engine = st.session_state.engine

# --- BARRA LATERAL (SETUP) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # 1. API
    api_key = st.text_input("1. Gemini API Key", type="password")
    if api_key and not engine.model:
        ok, msg = engine.configure_api(api_key)
        if ok: st.success(msg)
        else: st.error(msg)
    
    st.divider()
    
    # 2. CARGA DE LEY
    engine.entity = st.text_input("2. Entidad (Opcional)", placeholder="Ej: DIAN, Contraloría...")
    uploaded_text = st.text_area("3. Pegar Norma/Ley", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚠️ Borrar e Iniciar", type="primary"):
            if engine.process_law(uploaded_text, append=False) > 0:
                engine.simulacro_mode = False
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.toast("Entrenamiento iniciado", icon="🚀")
                st.rerun()
            else: st.error("Texto muy corto")
            
    with col2:
        if st.button("➕ Agregar Norma"):
            c = engine.process_law(uploaded_text, append=True)
            if c > 0: st.success(f"+{c} bloques.")
            else: st.error("Texto vacío")

    st.divider()
    
    # 3. MODOS Y GESTIÓN
    if st.button("🔥 MODO SIMULACRO", disabled=not engine.chunks):
        engine.simulacro_mode = True
        st.session_state.current_data = None
        st.session_state.page = 'game'
        st.toast("Modo Simulacro Activado", icon="🔥")
        st.rerun()

    st.divider()
    
    # 4. GUARDAR / CARGAR (JSON)
    st.markdown("### 💾 Persistencia")
    
    # Descargar
    if engine.chunks:
        data_to_save = {
            "chunks": engine.chunks,
            "mastery": {str(k):v for k,v in engine.mastery_tracker.items()},
            "failed": list(engine.failed_indices),
            "log": engine.mistakes_log,
            "feed": engine.feedback_history,
            "entity": engine.entity
        }
        json_str = json.dumps(data_to_save)
        st.download_button("Bajar Progreso", json_str, file_name="progreso_legal.json", mime="application/json")
    
    # Cargar
    uploaded_file = st.file_uploader("Subir Progreso", type=['json'])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            engine.chunks = data['chunks']
            engine.mastery_tracker = {int(k):v for k,v in data['mastery'].items()}
            engine.failed_indices = set(data['failed'])
            engine.mistakes_log = data['log']
            engine.feedback_history = data['feed']
            engine.entity = data.get('entity', "")
            st.success("¡Sesión restaurada!")
        except: st.error("Archivo inválido")

# --- PÁGINA PRINCIPAL ---

if st.session_state.page == 'setup':
    st.title("🏛️ Entrenador Legal PRO v3.6")
    st.info("👈 Configura tu API y carga la norma en la barra lateral para comenzar.")
    st.markdown("""
    ### Novedades:
    * **Anti-Bloqueo:** Espera automática si Google se satura.
    * **Simulacro:** Mezcla todas las normas cargadas.
    * **Maestría:** Sistema de repetición espaciada.
    """)

elif st.session_state.page == 'game':
    # HEADER
    perc, fails = engine.get_progress_stats()
    bar_color = "red" if engine.simulacro_mode else "blue"
    st.progress(perc/100, text=f"Progreso: {perc}% | Repasos pendientes: {fails} | Modo: {'SIMULACRO' if engine.simulacro_mode else 'ESTUDIO'}")

    # GENERACIÓN DE CASO
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        with st.spinner("⚖️ Analizando norma y redactando caso..."):
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

    # VISTA CASO
    with st.container(border=True):
        st.markdown(f"#### 📜 Historia del Caso")
        st.write(data['narrativa_caso'])
    
    if q_idx < len(data['preguntas']):
        q = data['preguntas'][q_idx]
        st.subheader(f"Pregunta {q_idx + 1}")
        st.markdown(f"**{q['enunciado']}**")

        # OPCIONES
        opts = q['opciones']
        opt_list = [f"A) {opts.get('A','')}", f"B) {opts.get('B','')}", f"C) {opts.get('C','')}"]
        
        # Formulario para evitar recargas
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
                    st.error(f"❌ Incorrecto. Era la {corr}.")
                    engine.failed_indices.add(engine.current_chunk_idx)
                    engine.mistakes_log.append({"pregunta": q['enunciado'], "elegida": sel, "correcta": corr})
                
                st.info(f"💡 {q['explicacion']}")
                st.session_state.answered = True

        # NAVEGACIÓN
        if st.session_state.answered:
            c1, c2 = st.columns([1, 1])
            with c1:
                lbl = "⏭️ Siguiente Pregunta" if q_idx < len(data['preguntas'])-1 else "🔄 Siguiente Caso"
                if st.button(lbl):
                    if q_idx < len(data['preguntas'])-1:
                        st.session_state.q_idx += 1
                    else:
                        # Fin del caso
                        if not engine.simulacro_mode and engine.current_chunk_idx not in engine.failed_indices:
                            engine.mastery_tracker[engine.current_chunk_idx] = engine.mastery_tracker.get(engine.current_chunk_idx, 0) + 1
                        st.session_state.current_data = None
                    st.session_state.answered = False
                    st.rerun()
            
            with c2:
                with st.expander("📢 Reportar Error / Calibrar"):
                    reason = st.selectbox("Motivo:", [
                        "repetitivo", "alucinacion", "error_estructural", 
                        "sesgo_longitud", "desconectado", "caso_simple", "pregunta_facil"
                    ])
                    if st.button("Enviar Reporte"):
                        engine.feedback_history.append(reason)
                        st.toast("Algoritmo ajustado.", icon="🛠️")

# --- REPORTE ---
if st.button("📊 Ver Reporte de Errores"):
    if not engine.mistakes_log:
        st.success("¡Sin errores por ahora!")
    else:
        st.write("### 📉 Historial de Fallos")
        for m in engine.mistakes_log:
            st.warning(f"**P:** {m['pregunta']} | **Tú:** {m['elegida']} | **Era:** {m['correcta']}")