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
    Esta función se ejecuta en cada recarga para limpiar y mantener la consistencia de los datos.
    1. Limpia recetas de productos que ya no existen.
    2. También limpia las recetas de recursos que ya no existen (insumos, equipos, personal).
    """
    if 'productos' in st.session_state and 'recetas' in st.session_state:
        productos_validos = set(st.session_state.productos['Nombre'].unique())
        insumos_validos = set(st.session_state.insumos['Nombre'].unique())
        equipos_validos = set(st.session_state.equipos['Nombre'].unique())
        personal_validos = set(st.session_state.personal['Rol'].unique())
        
        # Limpiar recetas de productos que ya no existen
        recetas_actuales = st.session_state.recetas
        recetas_limpias_productos = recetas_actuales[recetas_actuales['Producto'].isin(productos_validos)]
        
        # Limpiar recetas de recursos que ya no existen (Insumo, Equipo, Personal)
        recetas_limpias_recursos = recetas_limpias_productos[
            (recetas_limpias_productos['Tipo'] == 'Insumo') & (recetas_limpias_productos['Recurso'].isin(insumos_validos)) |
            (recetas_limpias_productos['Tipo'] == 'Equipo') & (recetas_limpias_productos['Recurso'].isin(equipos_validos)) |
            (recetas_limpias_productos['Tipo'] == 'Personal') & (recetas_limpias_productos['Recurso'].isin(personal_validos))
        ]
        
        st.session_state.recetas = recetas_limpias_recursos

def optimizar_produccion(productos, insumos, equipos, personal, recetas, params):
    num_productos = len(productos)
    if num_productos == 0: return None, "No se han definido productos para optimizar.", None, None, None
    
    # Asegurarse de que 'recetas' no esté vacío y tenga las columnas esperadas
    if recetas.empty: return None, "No se han definido recetas para los productos.", None, None, None
    if not all(col in recetas.columns for col in ['Producto', 'Tipo', 'Recurso', 'Cantidad']):
        return None, "El DataFrame de recetas no tiene las columnas esperadas.", None, None, None

    costo_insumos_por_producto, costo_personal_por_producto = [], []
    for i, prod in productos.iterrows():
        costo_i, costo_p = 0, 0
        receta_prod = recetas[recetas['Producto'] == prod['Nombre']]
        
        if receta_prod.empty:
            # Si un producto no tiene receta, su costo unitario es 0
            costo_insumos_por_producto.append(0)
            costo_personal_por_producto.append(0)
            continue # Pasar al siguiente producto

        for j, item_receta in receta_prod.iterrows():
            if item_receta['Tipo'] == 'Insumo':
                insumo_data = insumos[insumos['Nombre'] == item_receta['Recurso']]
                if not insumo_data.empty:
                    costo_insumo_unitario = insumo_data['Costo Unitario'].values[0]
                    costo_i += item_receta['Cantidad'] * costo_insumo_unitario
                else:
                    st.warning(f"Insumo '{item_receta['Recurso']}' en receta de '{prod['Nombre']}' no encontrado.")
            elif item_receta['Tipo'] == 'Personal':
                personal_data = personal[personal['Rol'] == item_receta['Recurso']]
                if not personal_data.empty:
                    costo_hora_personal = personal_data['Costo por Hora'].values[0]
                    costo_p += item_receta['Cantidad'] * costo_hora_personal
                else:
                    st.warning(f"Personal '{item_receta['Recurso']}' en receta de '{prod['Nombre']}' no encontrado.")
        costo_insumos_por_producto.append(costo_i)
        costo_personal_por_producto.append(costo_p)
    
    # Manejar el caso donde no hay recetas para algunos productos o datos faltantes
    if len(costo_insumos_por_producto) != num_productos:
        return None, "Error en el cálculo de costos unitarios de productos. Verifica tus recetas y recursos.", None, None, None

    precio_venta_neto = productos['Precio de Venta'].values * (1 - params['iibb'] / 100)
    beneficio_unitario = precio_venta_neto - np.array(costo_insumos_por_producto) - np.array(costo_personal_por_producto)
    c = -beneficio_unitario

    constraints_A, constraints_b = [], []

    # Restricciones de Insumos
    for _, insumo in insumos.iterrows():
        row = []
        for _, prod in productos.iterrows():
            cantidad_insumo_receta = recetas[(recetas['Producto'] == prod['Nombre']) & 
                                             (recetas['Recurso'] == insumo['Nombre']) & 
                                             (recetas['Tipo'] == 'Insumo')]['Cantidad'].sum()
            row.append(cantidad_insumo_receta)
        constraints_A.append(row)
        constraints_b.append(insumo['Cantidad Disponible'])

    # Restricciones de Equipos
    for _, equipo in equipos.iterrows():
        row = []
        for _, prod in productos.iterrows():
            cantidad_equipo_receta = recetas[(recetas['Producto'] == prod['Nombre']) & 
                                             (recetas['Recurso'] == equipo['Nombre']) & 
                                             (recetas['Tipo'] == 'Equipo')]['Cantidad'].sum()
            row.append(cantidad_equipo_receta)
        constraints_A.append(row)
        constraints_b.append(equipo['Horas Disponibles'])

    # Restricciones de Personal
    for _, p in personal.iterrows():
        row = []
        for _, prod in productos.iterrows():
            cantidad_personal_receta = recetas[(recetas['Producto'] == prod['Nombre']) & 
                                               (recetas['Recurso'] == p['Rol']) & 
                                               (recetas['Tipo'] == 'Personal')]['Cantidad'].sum()
            row.append(cantidad_personal_receta)
        constraints_A.append(row)
        constraints_b.append(p['Cantidad de Empleados'] * p['Horas por Empleado'])
    
    # Restricciones de Demanda Máxima
    for i, prod in productos.iterrows():
        row = np.zeros(num_productos)
        row[i] = 1
        constraints_A.append(row)
        constraints_b.append(prod['Demanda Máxima'])
        
    A_ub, b_ub = np.array(constraints_A), np.array(constraints_b)
    bounds = [(0, None) for _ in range(num_productos)]

    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    costos_variables = {'insumos': np.array(costo_insumos_por_producto), 'personal': np.array(costo_personal_por_producto)}
    
    if resultado.success: return resultado, None, A_ub, b_ub, costos_variables
    else: return None, resultado.message, None, None, None

def call_llama_api(api_key, context, question):
    if not api_key:
        return "Por favor, introduce tu API Key de Hugging Face."
    try:
        client = InferenceClient(token=api_key)
        
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
# Esta función ahora también limpia recursos huérfanos, no solo productos.
clean_up_data()

# --- Barra Lateral y Navegación ---
st.sidebar.header("Navegación")
page = st.sidebar.radio("Ir a:", ["⚙️ 1. Configuración de Recursos", "📝 2. Definición de Procesos", "📈 3. Parámetros Financieros", "🚀 4. Optimización y Resultados", "🧠 5. Análisis con IA"])
st.sidebar.header("🔑 Configuración API")
try:
    hf_api_key = st.secrets["HF_API_KEY"]
    st.sidebar.success("✅ API Key cargada desde Secrets.")
except:
    st.sidebar.warning("API Key no encontrada en Secrets. Para usar Llama 3.1, la API Key es obligatoria.")
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

    # Pre-calculamos las opciones para los selectbox individuales
    productos_validos = list(st.session_state.productos['Nombre'].unique())
    tipos_recurso_validos = ['Insumo', 'Equipo', 'Personal']

    # --- Sección para AGREGAR nuevas recetas ---
    st.subheader("Agregar Nueva Receta")
    with st.form("add_recipe_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_producto = st.selectbox("Producto", options=productos_validos, key="new_recipe_producto")
        with col2:
            new_tipo = st.selectbox("Tipo de Recurso", options=tipos_recurso_validos, key="new_recipe_tipo")
        
        recursos_disponibles = []
        if new_tipo == 'Insumo':
            recursos_disponibles = list(st.session_state.insumos['Nombre'].unique())
        elif new_tipo == 'Equipo':
            recursos_disponibles = list(st.session_state.equipos['Nombre'].unique())
        elif new_tipo == 'Personal':
            recursos_disponibles = list(st.session_state.personal['Rol'].unique())

        with col3:
            new_recurso = st.selectbox("Recurso Específico", options=recursos_disponibles, key="new_recipe_recurso", 
                                       disabled=not bool(recursos_disponibles)) # Deshabilitar si no hay recursos
        with col4:
            new_cantidad = st.number_input("Cantidad", min_value=0.01, value=1.0, step=0.1, key="new_recipe_cantidad")
        
        if st.form_submit_button("Añadir Receta"):
            if new_producto and new_tipo and new_recurso and new_cantidad > 0:
                # Comprobar si ya existe una receta idéntica
                if not st.session_state.recetas[(st.session_state.recetas['Producto'] == new_producto) &
                                                (st.session_state.recetas['Tipo'] == new_tipo) &
                                                (st.session_state.recetas['Recurso'] == new_recurso)].empty:
                    st.warning("Esta receta ya existe. Edítala en la tabla de abajo si quieres cambiar la cantidad.")
                else:
                    new_row = pd.DataFrame([{'Producto': new_producto, 'Tipo': new_tipo, 'Recurso': new_recurso, 'Cantidad': new_cantidad}])
                    st.session_state.recetas = pd.concat([st.session_state.recetas, new_row], ignore_index=True)
                    st.success(f"Receta para {new_producto} añadida.")
            else:
                st.error("Por favor, rellena todos los campos para añadir la receta.")
    
    st.divider()

    # --- Sección para EDITAR las recetas existentes ---
    st.subheader("Editar Recetas Existentes")
    # Mostrar el editor sin selectbox dinámicos en las columnas 'Producto', 'Tipo', 'Recurso'
    # Esto evitará los problemas de estado interno cuando los productos/recursos cambien
    st.session_state.recetas = st.data_editor(
        st.session_state.recetas,
        num_rows="dynamic",
        key="editor_recetas_final", # Un key fijo, ya que no tiene selectbox dinámicos
        column_config={
            "Producto": st.column_config.Column("Producto", help="Producto al que aplica esta receta"),
            "Tipo": st.column_config.Column("Tipo", help="Tipo de recurso (Insumo, Equipo, Personal)"),
            "Recurso": st.column_config.Column("Recurso", help="Nombre del insumo, equipo o rol de personal"),
            "Cantidad": st.column_config.NumberColumn("Cantidad", help="Cantidad necesaria por unidad de producto", min_value=0.01),
        }
    )

    # Nota: Los campos de Producto, Tipo y Recurso en el data_editor se muestran como texto
    # Si el usuario modifica el texto y este no existe en los recursos, la limpieza lo borrará o la optimización lo ignorará.

elif page == "📈 3. Parámetros Financieros":
    st.header("3. Parámetros Financieros y de Mercado")
    st.session_state.params['iibb'] = st.number_input("Tasa de Ingresos Brutos (%)", 0.0, 100.0, st.session_state.params.get('iibb', 3.5), 0.1)
    st.session_state.params['costo_capital'] = st.number_input("Costo de Capital / Financiero (%)", 0.0, 100.0, st.session_state.params.get('costo_capital', 8.0), 0.5)

elif page == "🚀 4. Optimización y Resultados":
    st.header("4. Optimización y Resultados")
    if st.button("▶️ Ejecutar Optimización", type="primary"):
        with st.spinner("Calculando..."):
            res, msg, A, b, costs = optimizar_produccion(st.session_state.productos, st.session_state.insumos, st.session_state.equipos, st.session_state.personal, st.session_state.recetas, st.session_state.params)
        if msg: 
            st.error(f"Error: {msg}")
            # Limpiar resultados anteriores si la optimización falla
            if 'resultados_optimizacion' in st.session_state:
                del st.session_state.resultados_optimizacion
                del st.session_state.produccion_optima
        else:
            st.success("¡Optimización completada!")
            st.session_state.resultados_optimizacion, st.session_state.A_ub, st.session_state.b_ub, st.session_state.costos_variables = res, A, b, costs
            st.session_state.produccion_optima = pd.DataFrame({'Producto': st.session_state.productos['Nombre'], 'Cantidad a Producir': res.x})
    
    if 'resultados_optimizacion' in st.session_state and st.session_state.resultados_optimizacion: # Verificar que haya resultados válidos
        res, costs = st.session_state.resultados_optimizacion, st.session_state.costos_variables
        
        # Calcular el beneficio bruto con el valor óptimo (fun es negativo)
        beneficio_bruto = -res.fun
        
        # Asegurarse de que los arrays de costos tengan la misma longitud que res.x
        if len(res.x) != len(costs['insumos']) or len(res.x) != len(costs['personal']):
            st.error("Error: La longitud de los resultados de producción no coincide con los costos. Esto puede indicar un problema con la configuración de recetas o productos.")
            beneficio_neto = 0 # Establecer a 0 para evitar más errores
        else:
            costo_total_variable = np.dot(res.x, costs['insumos']) + np.dot(res.x, costs['personal'])
            tasa_capital = st.session_state.params.get('costo_capital', 0) / 100
            costo_financiero = costo_total_variable * tasa_capital # El costo financiero se aplica a los costos variables
            beneficio_neto = beneficio_bruto - costo_financiero
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Beneficio Bruto Óptimo", f"${beneficio_bruto:,.2f}")
        c2.metric("Costo Financiero", f"${costo_financiero:,.2f}", delta=f"-{st.session_state.params.get('costo_capital', 0)}%", delta_color="inverse")
        c3.metric("Beneficio Neto Final", f"${beneficio_neto:,.2f}")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Plan de Producción")
            df = st.session_state.produccion_optima
            st.dataframe(df[df['Cantidad a Producir'] > 0.01].sort_values(by='Cantidad a Producir', ascending=False), use_container_width=True)
        with c2:
            st.subheader("Uso de Recursos")
            # Construir las etiquetas de las restricciones dinámicamente
            labels = []
            for _, insumo in st.session_state.insumos.iterrows(): labels.append(f"Insumo: {insumo['Nombre']}")
            for _, equipo in st.session_state.equipos.iterrows(): labels.append(f"Equipo: {equipo['Nombre']}")
            for _, p in st.session_state.personal.iterrows(): labels.append(f"Personal: {p['Rol']}")
            for _, prod in st.session_state.productos.iterrows(): labels.append(f"Demanda: {prod['Nombre']}")
            
            recursos_usados = st.session_state.A_ub @ res.x
            
            df_uso = pd.DataFrame({'Restricción': labels, 'Usado': recursos_usados, 'Disponible': st.session_state.b_ub})
            df_uso['Uso (%)'] = np.where(df_uso['Disponible'] > 0, (df_uso['Usado'] / df_uso['Disponible']) * 100, 0)
            
            # Ordenar por porcentaje de uso para ver cuellos de botella más fácilmente
            st.dataframe(df_uso.sort_values(by='Uso (%)', ascending=False), use_container_width=True)
            st.session_state.uso_recursos = df_uso

elif page == "🧠 5. Análisis con IA":
    st.header("5. Análisis con IA y Contexto de Mercado")
    if 'resultados_optimizacion' not in st.session_state: st.warning("Ejecuta la optimización primero para generar un análisis con IA.")
    elif not st.session_state.resultados_optimizacion: st.warning("La optimización no produjo resultados válidos. Verifica tus datos.")
    else:
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

        st.subheader("2. Generar Insight")
        
        # Asegurarse de que los arrays de costos tengan la misma longitud
        res = st.session_state.resultados_optimizacion
        costs = st.session_state.costos_variables
        
        if len(res.x) != len(costs['insumos']) or len(res.x) != len(costs['personal']):
            beneficio_neto = 0 # Valor por defecto si hay inconsistencia
            st.error("Error al calcular el Beneficio Neto para la IA: Inconsistencia en los datos de optimización.")
        else:
            beneficio_bruto = -res.fun
            costo_total_variable = np.dot(res.x, costs['insumos']) + np.dot(res.x, costs['personal'])
            tasa_capital = st.session_state.params.get('costo_capital', 0) / 100
            costo_financiero = costo_total_variable * tasa_capital
            beneficio_neto = beneficio_bruto - costo_financiero
        
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
