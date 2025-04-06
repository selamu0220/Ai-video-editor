import os
import sys
from pathlib import Path
import logging
import streamlit as st
import tempfile
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar dependencias necesarias de manera más robusta
def check_dependencies():
    try:
        from utils.video_handler import extract_frames, get_video_info
        from utils.gemini_video_analyzer import process_video_with_gemini, analyze_video_with_gemini
        return True
    except ImportError as e:
        st.error(f"Error importando dependencias: {str(e)}")
        logger.error(f"Error de importación: {e}")
        return False

# Asegurarse que el directorio raíz está en el path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Verificar dependencias antes de continuar
if not check_dependencies():
    st.error("Error crítico: No se pudieron cargar las dependencias necesarias")
    st.stop()

# Importar después de verificar
from utils.video_handler import extract_frames, get_video_info
from utils.gemini_video_analyzer import process_video_with_gemini, analyze_video_with_gemini

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asegurarse que el directorio raíz está en el path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Inicializar utils
try:
    from utils import ensure_dependencies
    ensure_dependencies()
except Exception as e:
    logger.error(f"Error inicializando dependencias: {e}")
    st.error("Error inicializando dependencias. Por favor, revise los logs.")
    sys.exit(1)

try:
    from utils.gemini_video_analyzer import install_system_dependencies, process_video_with_gemini, analyze_video_with_gemini
    
    # Intentar instalar dependencias del sistema si es necesario
    if not os.path.exists("/usr/lib/x86_64-linux-gnu/libGL.so.1"):
        logger.info("Instalando dependencias del sistema...")
        install_system_dependencies()
except Exception as e:
    st.error(f"Error inicializando el sistema: {str(e)}")
    sys.exit(1)

from utils.video_processor import get_video_info, get_video_thumbnail

# Set up page configuration
st.set_page_config(
    page_title="Red Creativa",
    page_icon="🧠",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Cargar CSS personalizado desde archivo
def load_css(file_path):
    with open(file_path, "r") as f:
        return f.read()

# Intentar cargar CSS desde assets
try:
    css = load_css("assets/styles.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception as e:
    print(f"No se pudo cargar el CSS: {e}")

# Custom CSS adicional para estilo similar a la imagen de referencia
st.markdown("""
<style>
    /* Estilo general */
    .stApp {
        background-color: #f5f3ef;
        color: #000000;
        font-family: 'Inter', sans-serif;
    }
    
    /* Estilos de encabezado */
    h1, h2, h3 {
        font-family: 'EB Garamond', serif;
        font-weight: 400;
    }
    
    h1 {
        font-size: 3.2rem !important;
        letter-spacing: -0.02em;
        line-height: 1.1 !important;
    }
    
    /* Contenedor principal */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Botones */
    .stButton > button {
        background-color: #000000;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #333333;
    }
    
    /* Elementos de entrada */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 1px solid #ddd;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    
    /* Estilos para secciones de contenido */
    .content-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Estilo para logo y título principal */
    .title-section {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Badge estilo */
    .badge {
        background-color: #e9e5e0;
        color: #333;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    /* Estilo para botón principal tipo CTA */
    .cta-button {
        background-color: #000;
        color: white;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 500;
        border: none;
        cursor: pointer;
        text-align: center;
        display: block;
        margin: 1rem auto;
        width: fit-content;
    }
    
    /* Estilo para resultados */
    .result-container {
        background-color: #f9f8f6;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 4px solid #000;
    }
    
    /* Ocultar elementos de Streamlit que no queremos */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Barra de desplazamiento */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f5f3ef;
    }
    ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #999;
    }
    
    /* Estilo para el archivo subido */
    .uploaded-file-info {
        background-color: #f9f8f6;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid #eee;
    }
    
    /* Navegación superior */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
    }
    
    .nav-logo {
        font-weight: bold;
        color: #000;
        text-decoration: none;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    
    .nav-link {
        color: #666;
        text-decoration: none;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de la sesión
if 'current_video_path' not in st.session_state:
    st.session_state.current_video_path = None

if 'video_info' not in st.session_state:
    st.session_state.video_info = None
    
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.TemporaryDirectory()
    
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
    
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Clase para cargar información del video en segundo plano
class VideoInfoLoader:
    def __init__(self):
        self.is_loading = False
        self.thread = None
    
    def load_info_async(self, video_path):
        if self.is_loading:
            return
        self.is_loading = True
        self.thread = threading.Thread(target=self._load_video_info, args=(video_path,))
        self.thread.daemon = True
        self.thread.start()
    
    def _load_video_info(self, video_path):
        try:
            # Cargar información real en segundo plano
            video_info = get_video_info(video_path)
            st.session_state.video_info = video_info
        except Exception as e:
            print(f"Error en carga asíncrona: {str(e)}")
        finally:
            self.is_loading = False

# Inicializar el cargador de información
if 'video_info_loader' not in st.session_state:
    st.session_state.video_info_loader = VideoInfoLoader()

# Función para formatear duración del video
def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(minutes)}m {int(seconds)}s"

# Navegación superior
st.markdown("""
<div class="top-nav">
    <a href="#" class="nav-logo">Red Creativa</a>
    <div class="nav-links">
        <a href="#" class="nav-link">Changelog</a>
    </div>
    <a href="#" class="nav-link">Request Access</a>
</div>
""", unsafe_allow_html=True)

# Header principal con estilo similar a la imagen de referencia
st.markdown('<div class="title-section">', unsafe_allow_html=True)
# Creando un logo simple
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <div style="display: inline-block; width: 40px; height: 40px; background-color: black; border-radius: 8px; margin-bottom: 10px;">
        <span style="color: white; font-size: 24px; line-height: 40px; font-family: 'Inter', sans-serif;">R</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<h1 class="main-heading">Red Creativa</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Automate high quality account research<br>to speed up your pipeline generation</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Descripción secundaria
st.markdown('<div style="text-align: center; margin-bottom: 3rem;">', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.1rem; max-width: 800px; margin: 0 auto;">Red Creativa analiza tus videos y genera insights que normalmente tardarías horas en descubrir.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Estructura de la aplicación en columnas
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">Sube tu video</h3>', unsafe_allow_html=True)
    
    # Selector de modelo de IA
    ai_models = {
        "Google Gemini": "Gemini Pro Vision para análisis visual avanzado"
    }
    
    selected_model = st.selectbox("Seleccionar modelo de IA:", 
                                 list(ai_models.keys()),
                                 index=0,
                                 format_func=lambda x: f"{x}")
    
    st.session_state.ai_model = selected_model
    
    # Subida de archivos
    uploaded_file = st.file_uploader("Sube un video para analizar", 
                                     type=["mp4", "mov", "avi", "mkv"],
                                     help="Formatos soportados: MP4, MOV, AVI, MKV")
    
    if uploaded_file:
        # Guardar el archivo subido
        temp_video_path = os.path.join(st.session_state.temp_dir.name, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Actualizar el video actual
        st.session_state.current_video_path = temp_video_path
        
        # Cargar información del video
        st.session_state.video_info_loader.load_info_async(temp_video_path)
        
        # Mostrar vista previa del video
        st.video(temp_video_path)
        
        # Mostrar información básica
        if st.session_state.video_info:
            info = st.session_state.video_info
            st.markdown('<div class="uploaded-file-info">', unsafe_allow_html=True)
            st.markdown(f"**Nombre:** {os.path.basename(temp_video_path)}")
            st.markdown(f"**Duración:** {format_duration(float(info.get('duration', 0)))}")
            st.markdown(f"**Resolución:** {info.get('width', '?')}x{info.get('height', '?')}")
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">Análisis de video con IA</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    Utiliza Google Gemini Vision para **analizar visualmente tu video**. 
    La IA puede identificar objetos, escenas, personas y generar instrucciones de edición basadas en el contenido.
    
    Para comenzar:
    1. Sube un video
    2. Escribe tu instrucción o consulta
    3. Haz clic en "Analizar con IA"
    """)
    
    # Campo para prompt del usuario
    user_prompt = st.text_area(
        "¿Qué quieres saber o hacer con este video?",
        height=100,
        placeholder="Ej: 'Analiza este video y dime qué efectos podrían mejorarlo' o 'Genera un comando para aplicar un efecto de cine'"
    )
    
    # Botón para procesar
    process_button = st.button(
        "Analizar con IA",
        key="vision_button",
        disabled=st.session_state.processing or not user_prompt or not st.session_state.current_video_path,
        help="Analiza el video con Google Gemini Vision"
    )
    
    # Procesar video si se hace clic en el botón
    if process_button and user_prompt and st.session_state.current_video_path:
        try:
            with st.spinner("Analizando video con IA... Esto puede tomar unos momentos."):
                st.session_state.processing = True
                
                # Llamar a la función de análisis con Gemini
                result = analyze_video_with_gemini(
                    st.session_state.current_video_path,
                    user_prompt,
                    num_frames=5,
                    temp_dir=st.session_state.temp_dir.name
                )
                
                st.session_state.analysis_result = result
        except Exception as e:
            st.error(f"Error al procesar: {str(e)}")
            traceback.print_exc()
        finally:
            st.session_state.processing = False
            st.rerun()
    
    # Mostrar resultados del análisis
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            with st.expander("Resultado del análisis", expanded=True):
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Mostrar análisis general
                if "análisis" in result:
                    st.subheader("Análisis")
                    st.write(result["análisis"])
                
                # Mostrar evaluación técnica
                if "evaluación_técnica" in result:
                    st.subheader("Evaluación técnica")
                    st.write(result["evaluación_técnica"])
                
                # Mostrar recomendación
                if "recomendación" in result:
                    st.subheader("Recomendación")
                    st.write(result["recomendación"])
                
                # Mostrar comandos FFmpeg
                if "comandos_ffmpeg" in result and result["comandos_ffmpeg"]:
                    st.subheader("Comandos FFmpeg sugeridos")
                    for i, cmd_info in enumerate(result["comandos_ffmpeg"]):
                        with st.expander(f"Comando {i+1}: {cmd_info.get('descripción', 'Sin descripción')}"):
                            st.code(cmd_info.get("comando", ""), language="bash")
                            
                            if "parámetros" in cmd_info:
                                st.write("Explicación de parámetros:")
                                for param, desc in cmd_info["parámetros"].items():
                                    st.markdown(f"- **{param}**: {desc}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Añadir botón de solicitud de acceso como en la imagen
st.markdown('<div style="text-align: center; margin-top: 3rem;">', unsafe_allow_html=True)
st.markdown('<a href="#" class="cta-button">Request Access</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
st.markdown("""
<div style='text-align: center; margin-top: 3rem; color: #999; font-size: 0.8rem;'>
    Red Creativa © 2023 - Análisis de video con inteligencia artificial
</div>
""", unsafe_allow_html=True)
