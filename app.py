import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re
from collections import Counter

# --- 1. CONFIGURACIÓN VISUAL ROBUSTA ---
st.set_page_config(page_title="Entrenador Legal TITÁN v5.0", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.02);}
    .narrative-box {
        background-color: #f8f9fa; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 6px solid #1f618d; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        font-family: 'Georgia', serif;
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

# --- 2. CEREBRO LÓGICO (RECUPERANDO LA COMPLEJIDAD INICIAL) ---
class LegalEngineTITAN:
    def __init__(self):
        # Memoria de Contenidos
        self.chunks = []           # Fragmentos de la ley
        self.chunk_origins = {}    # Para saber de qué ley vino cada fragmento (Multi-norma)
        
        # Memoria de Progreso (Algoritmo de Repetición Espaciada)
        self.mastery_tracker = {}  # {indice_chunk: nivel_maestria (0-3)}
        self.failed_indices = set()
        self.mistakes_log = []
        self.feedback_history = []
        
        # Estado Actual
        self.current_data = None
        self.current_chunk_idx = -1
        self.entity = ""
        self.simulacro_mode = False
        
        # Configuración IA
        self.model = None
        self.current_temperature = 0.2 # Temperatura baja para precisión jurídica

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Prioridad: Flash (Rápido) -> Pro (Potente)
            selected = next((m for m in models if 'gemini-1.5-flash' in m.lower()), None)
            if not selected: selected = next((m for m in models if 'flash' in m.lower()), models[0])
            self.model = genai.GenerativeModel(selected)
            return True, f"Conectado al Cerebro: {selected.split('/')[-1]}"
        except Exception as e:
            return False, f"Error Crítico API: {str(e)}"

    def process_law(self, text, append=False):
        """Procesa la ley cortándola en bloques lógicos para estudio profundo."""
        text = text.replace('\r', '')
        if len(text) < 100: return 0
        
        # Algoritmo de corte inteligente (evita cortar frases a la mitad si es posible)
        new_chunks = []
        step = 5500 # Tamaño óptimo para contexto de 4 preguntas
        for i in range(0, len(text), step):
            chunk = text[i:i+step]
            new_chunks.append(chunk)
            
        if not append:
            # Modo: Borrón y Cuenta Nueva
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
            self.mistakes_log = []
        else:
            # Modo: Acumulativo (Agregar Norma)
            start_index = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)):
                real_idx = start_index + i
                self.mastery_tracker[real_idx] = 0 # Inicia en maestría 0
        
        return len(new_chunks)

    def get_stats(self):
        if not self.chunks: return 0, 0, 0
        total_chunks = len(self.chunks)
        # Meta: Cada bloque debe aprobarse 3 veces (GOAL = 3)
        goal_score = total_chunks * 3 
        current_score = sum(self.mastery_tracker.values())
        
        percentage = int((current_score / goal_score) * 100) if goal_score > 0 else 0
        pending_reviews = len(self.failed_indices)
        
        return percentage, pending_reviews, total_chunks

    def get_calibration_prompt(self):
        """Genera las instrucciones de ajuste fino basadas en el feedback del usuario."""
        if not self.feedback_history: return "Modo: Estándar CNSC."
        
        counts = Counter(self.feedback_history)
        instructions = []
        
        # Lógica de Calibración Robusta
        if counts['respuesta_obvia'] > 0:
            instructions.append("💀 DIFICULTAD MÁXIMA: Los distractores (opciones falsas) deben ser SEMÁNTICAMENTE IDÉNTICOS a la correcta, variando solo un término técnico (plazo, autoridad, verbo rector).")
        
        if counts['pregunta_facil'] > 0:
            instructions.append("⚠️ TRAMPA DE ATENCIÓN: La respuesta correcta debe depender de un detalle fáctico minúsculo mencionado en la historia (una fecha específica, un cargo, una condición).")
        
        if counts['desconectado'] > 0:
            instructions.append("🔗 DEPENDENCIA OBLIGATORIA: Es prohibido hacer preguntas teóricas generales. El usuario DEBE leer el caso para responder.")
            
        if counts['repetitivo'] > 0:
            self.current_temperature = 0.6 # Subimos creatividad solo un poco
            instructions.append("🔄 VARIEDAD SITUACIONAL: Cambia radicalmente el escenario (Ej: Si antes fue contratación, ahora disciplinario; cambia nombres y cargos).")
        
        if counts['alucinacion'] > 0:
            self.current_temperature = 0.0 # Creatividad CERO
            instructions.append("⛔ FUENTE CERRADA ESTRICTA: Cita textual obligatoria. Si no está en el texto, no existe.")

        return "\n".join(instructions)

    def _safe_call(self, prompt):
        """Sistema Anti-Bloqueo de Google (Reintentos automáticos)."""
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
        
        # 1. ALGORITMO DE SELECCIÓN DE BLOQUE (INTELIGENTE)
        if self.simulacro_mode:
            # Simulacro: Caos total, cualquier bloque de cualquier ley cargada
            idx = random.choice(range(len(self.chunks)))
        else:
            # Estudio: Prioridad a lo fallado (Repetición Espaciada)
            if self.failed_indices:
                # 60% de probabilidad de que te salga algo que fallaste antes
                if random.random() < 0.6:
                    idx = random.choice(list(self.failed_indices))
                else:
                    # Si no, algo que te falte dominar
                    pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                    idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
            else:
                pending = [k for k,v in self.mastery_tracker.items() if v < 3]
                idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        text_chunk = self.chunks[idx]
        current_level = self.mastery_tracker.get(idx, 0)
        
        # 2. SISTEMA DE LENTES (COMPLEJIDAD PROGRESIVA)
        lentes = [
            "NIVEL 1: CONCEPTUAL (Definiciones y Naturaleza Jurídica)",
            "NIVEL 2: PROCEDIMENTAL (Trámites, Plazos y Competencias)",
            "NIVEL 3: SANCIONATORIO/EXCEPCIONES (Lo que rompe la regla)",
            "NIVEL 4: SITUACIONAL COMPLEJO (Aplicación e Integración)"
        ]
        lente_actual = lentes[min(current_level, 3)]
        
        contexto = f"CONTEXTO INSTITUCIONAL: {self.entity.upper()}" if self.entity else ""

        # 3. EL PROMPT "ROBUSTO" (RECUPERADO Y MEJORADO)
        prompt = f"""
        ACTÚA COMO UN EXPERTO DISEÑADOR DE PRUEBAS DE LA COMISIÓN NACIONAL DEL SERVICIO CIVIL (CNSC).
        
        TU INSUMO DE VERDAD (FUENTE CERRADA):
        ---------------------------------------------------------
        "{text_chunk[:6000]}"
        ---------------------------------------------------------
        
        TU MISIÓN:
        Diseñar un CASO SITUACIONAL riguroso con 4 PREGUNTAS de selección múltiple.
        
        PARÁMETROS DE DISEÑO:
        1.  **ENFOQUE COGNITIVO:** {lente_actual}.
        2.  **AMBIENTACIÓN:** {contexto} (Úsalo para dar "sabor" a los cargos, pero la respuesta jurídica sale estrictamente del texto).
        3.  **DEPENDENCIA DE LECTURA:** El caso debe tener detalles (fechas, cantidades, nombres de cargos) que sean INDISPENSABLES para responder. Si el usuario no lee el caso, debe fallar.
        4.  **ANTI-ALUCINACIÓN:** Cada respuesta correcta debe tener un respaldo explícito en el texto provisto.
        
        ESTRUCTURA OBLIGATORIA:
        * **Narrativa:** Un párrafo denso (150-200 palabras) describiendo una situación problemática en la administración pública.
        * **4 Preguntas:** Cada una debe evaluar un aspecto diferente del texto (un término, un procedimiento, una prohibición, una competencia).
        
        AJUSTES DE CALIBRACIÓN (USUARIO):
        {self.get_calibration_prompt()}
        
        FORMATO DE SALIDA (JSON PURO):
        {{
            "narrativa_caso": "El día 15 de marzo, en la entidad...",
            "preguntas": [
                {{
                    "enunciado": "¿Cuál fue la irregularidad cometida por el funcionario X respecto al plazo?",
                    "opciones": {{ "A": "...", "B": "...", "C": "..." }},
                    "respuesta": "A",
                    "explicacion": "Es A porque el texto dice: '...'"
                }},
                ... (hasta completar 4 preguntas)
            ]
        }}
        """
        
        res_json = self._safe_call(prompt)
        if not res_json: return {"error": "Error de conexión. Intenta de nuevo."}
        
        try:
            # Limpieza de JSON
            clean_text = res_json.strip()
            if "```" in clean_text:
                clean_text = re.search(r'```(?:json)?(.*?)```', clean_text, re.DOTALL)
                if clean_text: clean_text = clean_text.group(1).strip()
            return json.loads(clean_text)
        except:
            return {"error": "Error procesando la respuesta de la IA. Intenta de nuevo."}

# --- 3. INICIALIZACIÓN DE ESTADO ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False

engine = st.session_state.engine

# --- 4. INTERFAZ GRÁFICA (SETUP) ---
with st.sidebar:
    st.title("⚙️ Panel de Control")
    
    # API Key
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
    
    # Carga de Norma
    engine.entity = st.text_input("Entidad (Contexto):", placeholder="Ej: Fiscalía, DIAN...")
    txt_input = st.text_area("Texto de la Norma:", height=200, help="Pega aquí la ley, decreto o manual.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚠️ INICIAR NUEVO", type="primary"):
            c = engine.process_law(txt_input, append=False)
            if c: 
                st.session_state.page = 'game'
                st.session_state.current_data = None
                st.rerun()
            else: st.error("Texto vacío.")
            
    with col2:
        if st.button("➕ AGREGAR NORMA"):
            c = engine.process_law(txt_input, append=True)
            if c: st.success(f"+{c} Bloques agregados.")
            else: st.error("Texto vacío.")
            
    st.divider()
    
    # Simulacro
    if st.button("🔥 MODO SIMULACRO", disabled=not engine.chunks, help="Mezcla todas las normas cargadas"):
        engine.simulacro_mode = True
        st.session_state.current_data = None
        st.session_state.page = 'game'
        st.rerun()
        
    st.divider()
    
    # Persistencia
    st.markdown("### 💾 Guardar Progreso")
    if engine.chunks:
        save_data = json.dumps({
            "chunks": engine.chunks,
            "mastery": engine.mastery_tracker,
            "failed": list(engine.failed_indices),
            "log": engine.mistakes_log,
            "feed": engine.feedback_history,
            "entity": engine.entity
        })
        st.download_button("Descargar Archivo .JSON", save_data, "progreso_titan.json", "application/json")
    
    upl = st.file_uploader("Cargar Archivo .JSON", type=['json'])
    if upl:
        try:
            d = json.load(upl)
            engine.chunks = d['chunks']
            # Convertir claves de string a int (JSON guarda keys como str)
            engine.mastery_tracker = {int(k):v for k,v in d['mastery'].items()}
            engine.failed_indices = set(d['failed'])
            engine.mistakes_log = d['log']
            engine.feedback_history = d['feed']
            engine.entity = d.get('entity', "")
            st.success("¡Sesión Recuperada!")
        except: st.error("Archivo corrupto.")

# --- 5. INTERFAZ PRINCIPAL (JUEGO) ---
if st.session_state.page == 'game':
    # Barra Superior
    perc, fails, total = engine.get_stats()
    mode_str = "🔥 SIMULACRO" if engine.simulacro_mode else "📚 ESTUDIO"
    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; background:#eee; padding:10px; border-radius:8px;'>
        <span class='status-bar'>MODO: {mode_str}</span>
        <span class='status-bar'>DOMINIO: {perc}%</span>
        <span class='status-bar'>BLOQUES: {total}</span>
        <span class='status-bar' style='color:red'>REPASOS: {fails}</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(perc/100)

    # Generación de Caso
    if not st.session_state.current_data:
        with st.spinner("⚖️ El Examinador está diseñando un caso complejo (4 preguntas)..."):
            data = engine.generate_case()
            if "error" in data:
                st.error(data['error'])
                if st.button("Reintentar Conexión"): st.rerun()
                st.stop()
            st.session_state.current_data = data
            st.session_state.q_idx = 0
            st.session_state.answered = False
            st.rerun()

    data = st.session_state.current_data
    q_idx = st.session_state.q_idx
    questions = data.get('preguntas', [])

    # Mostrar Narrativa (Siempre visible para obligar a leer)
    st.markdown(f"""
    <div class="narrative-box">
        <h4>📜 Caso Situacional</h4>
        <p style="font-size:1.1em; line-height:1.6;">{data.get('narrativa_caso', 'Error narrativo.')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Área de Pregunta
    if q_idx < len(questions):
        q = questions[q_idx]
        
        with st.container():
            st.markdown(f"### 🔹 Pregunta {q_idx + 1} de {len(questions)}")
            st.markdown(f"##### {q['enunciado']}")
            
            opts = q['opciones']
            # Aleatorizar orden visual si se quiere (opcional), aquí lo dejamos fijo A,B,C
            op_list = [f"A) {opts.get('A','')}", f"B) {opts.get('B','')}", f"C) {opts.get('C','')}"]
            
            with st.form(key=f"q_form_{q_idx}"):
                selection = st.radio("Selecciona tu respuesta:", op_list, index=None)
                submit = st.form_submit_button("✅ Validar Respuesta")
                
                if submit and selection:
                    letter = selection[0] # 'A', 'B' o 'C'
                    correct = q['respuesta'].upper()
                    
                    if letter == correct:
                        st.success("✅ ¡CORRECTO!")
                        # Si estaba en lista de fallos, lo sacamos (solo si era un repaso)
                        if engine.current_chunk_idx in engine.failed_indices:
                            engine.failed_indices.remove(engine.current_chunk_idx)
                    else:
                        st.error(f"❌ INCORRECTO. La respuesta era la {correct}.")
                        # Castigo: Se agrega a lista de fallos para repasar luego
                        engine.failed_indices.add(engine.current_chunk_idx)
                        engine.mistakes_log.append({
                            "caso": data['narrativa_caso'][:50]+"...",
                            "pregunta": q['enunciado'],
                            "error": letter,
                            "correcta": correct
                        })
                    
                    st.info(f"💡 **Fundamento Jurídico:** {q['explicacion']}")
                    st.session_state.answered = True

        # Navegación
        if st.session_state.answered:
            col_nav, col_rep = st.columns([1, 1])
            
            with col_nav:
                if q_idx < len(questions) - 1:
                    if st.button("⏭️ Siguiente Pregunta del Caso"):
                        st.session_state.q_idx += 1
                        st.session_state.answered = False
                        st.rerun()
                else:
                    if st.button("🔄 TERMINAR CASO (Generar Nuevo)"):
                        # Si aprobó todo el caso y no está en modo simulacro, sube maestría
                        if not engine.simulacro_mode:
                            idx = engine.current_chunk_idx
                            engine.mastery_tracker[idx] = engine.mastery_tracker.get(idx, 0) + 1
                        
                        st.session_state.current_data = None
                        st.session_state.q_idx = 0
                        st.session_state.answered = False
                        st.rerun()
            
            with col_rep:
                with st.expander("📢 Calibrar al Examinador (Reportar)"):
                    # Opciones de calibración completas
                    reasons = {
                        "Respuesta muy Obvia (Regalada)": "respuesta_obvia",
                        "Pregunta muy Fácil": "pregunta_facil",
                        "Caso Desconectado del Texto": "desconectado",
                        "Caso Repetitivo": "repetitivo",
                        "Alucinación (Inventó Norma)": "alucinacion"
                    }
                    selected_reason = st.selectbox("Motivo:", list(reasons.keys()))
                    if st.button("Enviar Feedback"):
                        engine.feedback_history.append(reasons[selected_reason])
                        st.toast("Algoritmo Recalibrado.", icon="🧠")

elif st.session_state.page == 'setup':
    st.markdown("""
    <div style="text-align:center; margin-top:50px;">
        <h1>🏛️ Entrenador Legal TITÁN v5.0</h1>
        <p>Carga tus normas en el menú lateral para iniciar.</p>
        <p><i>Características: Fuente Cerrada, 4 Preguntas/Caso, Anti-Bloqueo Google.</i></p>
    </div>
    """, unsafe_allow_html=True)