import streamlit as st
import google.generativeai as genai
import json
import random
import time
import re

# --- 1. CONFIGURACIÓN VISUAL (ESTILO v6.4 RESTAURADO) ---
st.set_page_config(page_title="TITÁN v8.0 - Estabilidad Total", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; height: 3.5em; transition: all 0.3s;}
    .narrative-box {
        background-color: #e8f5e9; padding: 25px; border-radius: 12px; 
        border-left: 6px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px; font-family: 'Georgia', serif; font-size: 1.15em;
    }
    .status-bar {font-weight: bold; color: #2e86c1;}
</style>
""", unsafe_allow_html=True)

# --- 2. MOTOR LÓGICO ---
class LegalEngineTITAN:
    def __init__(self):
        self.chunks = []
        self.mastery_tracker = {}
        self.failed_indices = set()
        self.entity = ""
        self.level = "Profesional"
        self.model = None

    def configure_api(self, key):
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            return True, "Conectado"
        except Exception as e:
            return False, str(e)

    def process_law(self, text, append=False):
        text = text.replace('\r', '')
        if len(text) < 50: return 0
        new_chunks = [text[i:i+6000] for i in range(0, len(text), 6000)]
        if not append:
            self.chunks = new_chunks
            self.mastery_tracker = {i: 0 for i in range(len(self.chunks))}
            self.failed_indices = set()
        else:
            start = len(self.chunks)
            self.chunks.extend(new_chunks)
            for i in range(len(new_chunks)): self.mastery_tracker[start+i] = 0
        return len(new_chunks)

    def generate_case(self):
        if not self.chunks: return None
        # Selección: 60% probabilidad de repetir un fallo previo
        if self.failed_indices and random.random() < 0.6:
            idx = random.choice(list(self.failed_indices))
        else:
            pending = [k for k,v in self.mastery_tracker.items() if v < 3]
            idx = random.choice(pending) if pending else random.choice(range(len(self.chunks)))
        
        self.current_chunk_idx = idx
        
        prompt = f"""
        ACTÚA COMO EXPERTO JURISTA CNSC. NIVEL: {self.level.upper()}.
        ESCENARIO: {self.entity.upper() if self.entity else 'GENERAL'}.
        TEXTO NORMATIVO: "{self.chunks[idx]}"
        
        TAREA: Crea un caso situacional con 4 preguntas. 
        REGLA DE ORO: Las opciones incorrectas deben ser leyes reales pero inaplicables a los hechos.
        Responde exclusivamente en JSON:
        {{
            "narrativa": "Caso detallado...",
            "preguntas": [
                {{"enunciado": "..", "opciones": {{"A": "..", "B": "..", "C": ".."}}, "respuesta": "A", "explicacion": ".."}}
            ]
        }}
        """
        try:
            res = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(res.text)
        except: return None

# --- 3. FLUJO DE NAVEGACIÓN ---
if 'engine' not in st.session_state: st.session_state.engine = LegalEngineTITAN()
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'answered' not in st.session_state: st.session_state.answered = False
engine = st.session_state.engine

# --- 4. PANEL LATERAL ---
with st.sidebar:
    st.title("⚖️ TITÁN v8.0")
    key = st.text_input("API Key:", type="password")
    if key and not engine.model:
        ok, msg = engine.configure_api(key)
        if ok: st.success("Conexión Estable")

    st.divider()
    engine.level = st.selectbox("Nivel:", ["Asistencial", "Técnico", "Profesional", "Asesor"], index=2)
    engine.entity = st.text_input("Entidad:", placeholder="Ej: Fiscalía General")
    txt = st.text_area("Cargar Norma:", height=200)
    
    if st.button("🚀 INICIAR ENTRENAMIENTO", type="primary"):
        if engine.process_law(txt):
            st.session_state.data = None
            st.session_state.page = 'game'
            st.rerun()

# --- 5. INTERFAZ DE JUEGO ---
if st.session_state.page == 'game':
    # Barra de estadísticas
    done = sum([min(v, 3) for v in engine.mastery_tracker.values()])
    total = len(engine.chunks) * 3
    perc = int((done/total)*100) if total > 0 else 0
    
    st.markdown(f"**DOMINIO: {perc}%** | **BLOQUES: {len(engine.chunks)}** | **REPASOS: {len(engine.failed_indices)}**")
    st.progress(perc/100)

    if not st.session_state.get('data'):
        with st.spinner("⚖️ IA analizando y diseñando caso..."):
            st.session_state.data = engine.generate_case()
            st.session_state.q_idx = 0
            st.session_state.answered = False
            st.rerun()

    data = st.session_state.data
    if data:
        st.markdown(f"<div class='narrative-box'><h4>📜 Caso: {engine.entity if engine.entity else 'General'}</h4>{data.get('narrativa', '')}</div>", unsafe_allow_html=True)
        
        q = data['preguntas'][st.session_state.q_idx]
        st.subheader(f"Pregunta {st.session_state.q_idx + 1}")
        st.write(q['enunciado'])
        
        with st.form(key=f"q_{st.session_state.q_idx}"):
            sel = st.radio("Opciones:", [f"{k}) {v}" for k,v in q['opciones'].items()], index=None)
            if st.form_submit_button("✅ VALIDAR RESPUESTA"):
                if sel and sel[0] == q['respuesta']:
                    st.success("¡Excelente! Es correcto.")
                    if engine.current_chunk_idx in engine.failed_indices: engine.failed_indices.remove(engine.current_chunk_idx)
                else:
                    st.error(f"Incorrecto. La respuesta era la {q['respuesta']}")
                    engine.failed_indices.add(engine.current_chunk_idx)
                st.info(q['explicacion'])
                st.session_state.answered = True

        if st.session_state.answered:
            if st.button("Siguiente ⏭️") if st.session_state.q_idx < 3 else st.button("🔄 FINALIZAR CASO"):
                if st.session_state.q_idx < 3:
                    st.session_state.q_idx += 1
                    st.session_state.answered = False
                    st.rerun()
                else:
                    engine.mastery_tracker[engine.current_chunk_idx] += 1
                    st.session_state.data = None
                    st.rerun()

elif st.session_state.page == 'setup':
    st.title("🏛️ Entrenador Legal TITÁN v8.0")
    st.write("Configura la norma en el panel izquierdo para comenzar.")