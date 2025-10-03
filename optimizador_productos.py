import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Optimizador de Rentabilidad Empresarial",
    page_icon="💰",
    layout="wide"
)

# --- Funciones Auxiliares ---

def clean_up_data():
    """
    ### SOLUCIÓN DEFINITIVA AL ValueError ###
    Esta función se ejecuta en cada recarga para asegurar la consistencia de los datos.
    Elimina recetas de productos que ya no existen.
    """
    if 'productos' in st.session_state and 'recetas' in st.session_state:
        productos_validos = st.session_state.productos['Nombre'].unique()
        recetas_actuales = st.session_state.recetas
        recetas_limpias = recetas_actuales[recetas_actuales['Producto'].isin(productos_validos)]
        st.session_state.recetas = recetas_limpias

def optimizar_produccion(productos, insumos, equipos, personal, recetas, params):
    # (Esta función ya es robusta y no necesita cambios)
    num_productos = len(productos)
    if num_productos == 0: return None, "No se han definido productos para optimizar.", None, None, None
    costo_insumos_por_producto, costo_personal_por_producto = [], []
    for i, prod in productos.iterrows():
        costo_i, costo_p = 0, 0
        receta_prod = recetas[recetas['Producto'] == prod['Nombre']]
        for j, item_receta in receta_prod.iterrows():
            if item_receta['Tipo'] == 'Insumo':
                costo_insumo_unitario = insumos[insumos['Nombre'] == item_receta['Recurso']]['Costo Unitario'].values[0]
                costo_i += item_receta['Cantidad'] * costo_insumo_unitario
            elif item_receta['Tipo'] == 'Personal':
                costo_hora_personal = personal[personal['Rol'] == item_receta['Recurso']]['Costo por Hora'].values[0]
                costo_p += item_receta['Cantidad'] * costo_hora_personal
        costo_insumos_por_producto.append(costo_i)
        costo_personal_por_producto.append(costo_p)
    precio_venta_neto = productos['Precio de Venta'].values * (1 - params['iibb'] / 100)
    beneficio_unitario = precio_venta_neto - np.array(costo_insumos_por_producto) - np.array(costo_personal_por_producto)
    c = -beneficio_unitario
    constraints_A, constraints_b = [], []
    for _, insumo in insumos.iterrows():
        row = [recetas[(recetas['Producto'] == prod['Nombre']) & (recetas['Recurso'] == insumo['Nombre']) & (recetas['Tipo'] == 'Insumo')]['Cantidad'].sum() for _, prod in productos.iterrows()]
        constraints_A.append(row); constraints_b.append(insumo['Cantidad Disponible'])
    for _, equipo in equipos.iterrows():
        row = [recetas[(recetas['Producto'] == prod['Nombre']) & (recetas['Recurso'] == equipo['Nombre']) & (recetas['Tipo'] == 'Equipo')]['Cantidad'].sum() for _, prod in productos.iterrows()]
        constraints_A.append(row); constraints_b.append(equipo['Horas Disponibles'])
    for _, p in personal.iterrows():
        row = [recetas[(recetas['Producto'] == prod['Nombre']) & (recetas['Recurso'] == p['Rol']) & (recetas['Tipo'] == 'Personal')]['Cantidad'].sum() for _, prod in productos.iterrows()]
        constraints_A.append(row); constraints_b.append(p['Cantidad de Empleados'] * p['Horas por Empleado'])
    for i, prod in productos.iterrows():
        row = np.zeros(num_productos); row[i] = 1
        constraints_A.append(row); constraints_b.append(prod['Demanda Máxima'])
    A_ub, b_ub = np.array(constraints_A), np.array(constraints_b)
    bounds = [(0, None) for _ in range(num_productos)]
    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    costos_variables = {'insumos': np.array(costo_insumos_por_producto), 'personal': np.array(costo_personal_por_producto)}
    if resultado.success: return resultado, None, A_ub, b_ub, costos_variables
    else: return None, resultado.message, None, None, None

def call_llama_api(api_key, context, question):
    """
    ### SOLUCIÓN DEFINITIVA API ###
    Usa el modelo Llama 3.1 con su plantilla de prompt oficial.
    """
    if not api_key:
        return "Por favor, introduce tu API Key de Hugging Face."
    try:
        client = InferenceClient(token=api_key)
        
        # Plantilla oficial para Llama 3.1 Instruct
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Eres un consultor de negocios experto. Analiza el contexto proporcionado que incluye datos de optimización y un análisis de mercado. Responde la pregunta del usuario de forma clara, concisa y ofreciendo recomendaciones accionables.<|eot_id|><|start_header_id|>user<|end_header_id|>

**Contexto:**
{context}

**Pregunta:**
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        response = client.text_generation(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        return response
    except Exception as e:
        return f"Error al contactar la API de Hugging Face: {e}"

# --- Interfaz de la App ---
st.title("💰 Optimizador de Rentabilidad Empresarial")

# --- Inicialización de Datos ---
if 'productos' not in st.session_state:
    st.session_state.productos = pd.DataFrame({'Nombre': ['Producto A', 'Producto B'], 'Demanda Máxima': [100.0, 150.0], 'Precio de Venta': [50.0, 75.0]})
# ... (resto de inicializaciones)
if 'insumos' not in st.session_state:
    st.session_state.insumos = pd.DataFrame({'Nombre': ['Insumo X', 'Insumo Y'], 'Cantidad Disponible': [500.0, 800.0], 'Costo Unitario': [5.0, 8.0]})
if 'equipos' not in st.session_state:
    st.session_state.equipos = pd.DataFrame({'Nombre': ['Máquina 1', 'Máquina 2'], 'Horas Disponibles': [40.0, 30.0]})
if 'personal' not in st.session_state:
    st.session_state.personal = pd.DataFrame({'Rol': ['Operario', 'Supervisor'], 'Cantidad de Empleados': [2, 1], 'Horas por Empleado': [40, 40], 'Costo por Hora': [15.0, 25.0]})
if 'recetas' not in st.session_state:
    st.session_state.recetas = pd.DataFrame({
        'Producto': ['Producto A', 'Producto A', 'Producto A', 'Producto B', 'Producto B', 'Producto B'],
        'Tipo': ['Insumo', 'Equipo', 'Personal', 'Insumo', 'Equipo', 'Personal'],
        'Recurso': ['Insumo X', 'Máquina 1', 'Operario', 'Insumo Y', 'Máquina 2', 'Operario'],
        'Cantidad': [2.0, 0.5, 1.0, 3.0, 0.2, 1.5]
    })
if 'params' not in st.session_state:
    st.session_state.params = {'iibb': 3.5, 'costo_capital': 8.0}

# --- Limpieza de Datos en cada Rerun ---
clean_up_data()

# --- Barra Lateral y Navegación ---
st.sidebar.header("Navegación")
page = st.sidebar.radio("Ir a:", ["⚙️ 1. Configuración de Recursos", "📝 2. Definición de Procesos", "📈 3. Parámetros Financieros", "🚀 4. Optimización y Resultados", "🧠 5. Análisis con IA"])
st.sidebar.header("🔑 Configuración API")
try:
    hf_api_key = st.secrets["HF_API_KEY"]
    st.sidebar.success("✅ API Key cargada desde Secrets.")
except:
    st.sidebar.warning("API Key no encontrada en Secrets.")
    hf_api_key = st.sidebar.text_input("Ingresa tu Hugging Face API Key", type="password")

# --- Contenido de las Páginas ---
if page == "⚙️ 1. Configuración de Recursos":
    st.header("1. Configuración de Recursos")
    st.subheader("A. Productos o Servicios")
    st.data_editor(st.session_state.productos, num_rows="dynamic", key="productos")
    st.subheader("B. Insumos / Materias Primas")
    st.data_editor(st.session_state.insumos, num_rows="dynamic", key="insumos")
    st.subheader("C. Equipos / Maquinaria")
    st.data_editor(st.session_state.equipos, num_rows="dynamic", key="equipos")
    st.subheader("D. Personal")
    st.data_editor(st.session_state.personal, num_rows="dynamic", key="personal")

elif page == "📝 2. Definición de Procesos":
    st.header("2. Definición de Procesos (Recetas)")
    productos_validos = st.session_state.productos['Nombre'].unique()
    st.data_editor(
        st.session_state.recetas,
        num_rows="dynamic",
        key="editor_recetas",
        column_config={
            "Producto": st.column_config.SelectboxColumn("Producto", options=productos_validos, required=True),
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=['Insumo', 'Equipo', 'Personal'], required=True),
            "Recurso": st.column_config.SelectboxColumn("Recurso", options=pd.concat([
                st.session_state.insumos['Nombre'], st.session_state.equipos['Nombre'], st.session_state.personal['Rol']]).unique(), required=True),
        }
    )

elif page == "📈 3. Parámetros Financieros":
    st.header("3. Parámetros Financieros y de Mercado")
    st.session_state.params['iibb'] = st.number_input("Tasa de Ingresos Brutos (%)", 0.0, 100.0, st.session_state.params.get('iibb', 3.5), 0.1)
    st.session_state.params['costo_capital'] = st.number_input("Costo de Capital / Financiero (%)", 0.0, 100.0, st.session_state.params.get('costo_capital', 8.0), 0.5)

elif page == "🚀 4. Optimización y Resultados":
    st.header("4. Optimización y Resultados")
    if st.button("▶️ Ejecutar Optimización", type="primary"):
        with st.spinner("Calculando..."):
            res, msg, A, b, costs = optimizar_produccion(st.session_state.productos, st.session_state.insumos, st.session_state.equipos, st.session_state.personal, st.session_state.recetas, st.session_state.params)
        if msg: st.error(f"Error: {msg}")
        else:
            st.success("¡Optimización completada!")
            st.session_state.resultados_optimizacion, st.session_state.A_ub, st.session_state.b_ub, st.session_state.costos_variables = res, A, b, costs
            st.session_state.produccion_optima = pd.DataFrame({'Producto': st.session_state.productos['Nombre'], 'Cantidad a Producir': res.x})
    if 'resultados_optimizacion' in st.session_state:
        # (código para mostrar resultados sin cambios)
        res, costs = st.session_state.resultados_optimizacion, st.session_state.costos_variables
        beneficio_bruto = -res.fun
        costo_total = np.dot(res.x, costs['insumos']) + np.dot(res.x, costs['personal'])
        tasa_capital = st.session_state.params.get('costo_capital', 0) / 100
        costo_financiero = costo_total * tasa_capital
        beneficio_neto = beneficio_bruto - costo_financiero
        c1, c2, c3 = st.columns(3); c1.metric("Beneficio Bruto Óptimo", f"${beneficio_bruto:,.2f}"); c2.metric("Costo Financiero", f"${costo_financiero:,.2f}", delta=f"-{st.session_state.params.get('costo_capital', 0)}%", delta_color="inverse"); c3.metric("Beneficio Neto Final", f"${beneficio_neto:,.2f}"); st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Plan de Producción"); df = st.session_state.produccion_optima; st.dataframe(df[df['Cantidad a Producir'] > 0.01], use_container_width=True)
        with c2:
            st.subheader("Uso de Recursos"); recursos_usados = st.session_state.A_ub @ res.x
            labels = [f"Insumo: {n}" for n in st.session_state.insumos['Nombre']] + [f"Equipo: {n}" for n in st.session_state.equipos['Nombre']] + [f"Personal: {n}" for n in st.session_state.personal['Rol']] + [f"Demanda: {n}" for n in st.session_state.productos['Nombre']]
            df_uso = pd.DataFrame({'Restricción': labels, 'Usado': recursos_usados, 'Disponible': st.session_state.b_ub}); df_uso['Uso (%)'] = np.where(df_uso['Disponible'] > 0, (df_uso['Usado'] / df_uso['Disponible']) * 100, 0)
            st.dataframe(df_uso, use_container_width=True); st.session_state.uso_recursos = df_uso

elif page == "🧠 5. Análisis con IA":
    st.header("5. Análisis con IA y Contexto de Mercado")
    if 'resultados_optimizacion' not in st.session_state: st.warning("Ejecuta la optimización primero.")
    else:
        # ### NUEVA FUNCIONALIDAD: CARGA DE PDF ###
        st.subheader("1. (Opcional) Cargar Archivo de Contexto")
        uploaded_file = st.file_uploader("Sube un PDF con análisis de mercado, precios de competidores, etc.", type="pdf")
        market_context = ""
        if uploaded_file is not None:
            try:
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                for page in pdf_reader.pages:
                    market_context += page.extract_text() + "\n"
                st.success("PDF procesado con éxito.")
                with st.expander("Ver texto extraído del PDF"):
                    st.text_area("", market_context, height=200)
            except Exception as e:
                st.error(f"Error al leer el PDF: {e}")

        # Construcción del contexto para la IA
        st.subheader("2. Generar Insight")
        beneficio_neto = -st.session_state.resultados_optimizacion.fun - (np.dot(st.session_state.resultados_optimizacion.x, st.session_state.costos_variables['insumos']) + np.dot(st.session_state.resultados_optimizacion.x, st.session_state.costos_variables['personal'])) * (st.session_state.params.get('costo_capital', 0) / 100)
        
        contexto_interno = f"Resultados de Optimización:\n- Beneficio Neto Final: ${beneficio_neto:,.2f}\n\nProducción Óptima:\n{st.session_state.produccion_optima.to_string()}\n\nUso de Recursos (Cuellos de Botella):\n{st.session_state.uso_recursos.to_string()}"
        
        contexto_completo = contexto_interno
        if market_context:
            contexto_completo += f"\n\nAnálisis de Mercado (del PDF):\n{market_context}"

        st.text_area("Contexto final enviado a la IA:", contexto_completo, height=300)
        pregunta = st.text_input("Haz tu pregunta:", "Basado en los datos de optimización y el contexto de mercado, ¿cuál debería ser mi principal foco estratégico?")
        
        if st.button("Obtener Insight con Llama 3.1", type="primary"):
            if not hf_api_key: st.error("Configura la API Key en la barra lateral.")
            else:
                with st.spinner("Llama 3.1 está pensando..."):
                    respuesta = call_llama_api(hf_api_key, contexto_completo, pregunta)
                    st.success("Análisis recibido:")
                    st.markdown(respuesta)
