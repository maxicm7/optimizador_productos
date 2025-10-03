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

def get_empty_dataframes():
    """Retorna DataFrames iniciales vacíos con las columnas correctas."""
    return {
        'productos': pd.DataFrame({'Nombre': pd.Series(dtype='str'), 'Demanda Máxima': pd.Series(dtype='float'), 'Precio de Venta': pd.Series(dtype='float')}),
        'insumos': pd.DataFrame({'Nombre': pd.Series(dtype='str'), 'Cantidad Disponible': pd.Series(dtype='float'), 'Costo Unitario': pd.Series(dtype='float')}),
        'equipos': pd.DataFrame({'Nombre': pd.Series(dtype='str'), 'Horas Disponibles': pd.Series(dtype='float')}),
        'personal': pd.DataFrame({'Rol': pd.Series(dtype='str'), 'Cantidad de Empleados': pd.Series(dtype='int'), 'Horas por Empleado': pd.Series(dtype='float'), 'Costo por Hora': pd.Series(dtype='float')}),
        'recetas': pd.DataFrame({'Producto': pd.Series(dtype='str'), 'Tipo': pd.Series(dtype='str'), 'Recurso': pd.Series(dtype='str'), 'Cantidad': pd.Series(dtype='float')}),
        'params': {'iibb': 3.5, 'costo_capital': 8.0}
    }

def clean_up_data():
    """
    Esta función se ejecuta en cada recarga para limpiar y mantener la consistencia de los datos.
    Elimina recetas que hacen referencia a productos o recursos (insumos, equipos, personal) que ya no existen.
    """
    if 'productos' in st.session_state and 'recetas' in st.session_state:
        productos_validos_set = set(st.session_state.productos['Nombre'].unique())
        insumos_validos_set = set(st.session_state.insumos['Nombre'].unique())
        equipos_validos_set = set(st.session_state.equipos['Nombre'].unique())
        personal_validos_set = set(st.session_state.personal['Rol'].unique())
        
        recetas_actuales = st.session_state.recetas
        
        # Filtrar por productos válidos
        recetas_limpias_productos = recetas_actuales[recetas_actuales['Producto'].isin(productos_validos_set)]
        
        # Filtrar por recursos válidos según su tipo
        filtered_recetas = []
        for index, row in recetas_limpias_productos.iterrows():
            is_valid = False
            if row['Tipo'] == 'Insumo' and row['Recurso'] in insumos_validos_set:
                is_valid = True
            elif row['Tipo'] == 'Equipo' and row['Recurso'] in equipos_validos_set:
                is_valid = True
            elif row['Tipo'] == 'Personal' and row['Recurso'] in personal_validos_set:
                is_valid = True
            
            if is_valid:
                filtered_recetas.append(row)
        
        st.session_state.recetas = pd.DataFrame(filtered_recetas, columns=recetas_actuales.columns)


def optimizar_produccion(productos, insumos, equipos, personal, recetas, params):
    num_productos = len(productos)
    if num_productos == 0: return None, "No se han definido productos para optimizar.", None, None, None
    if recetas.empty: return None, "No se han definido recetas para los productos.", None, None, None
    
    # Asegurarse de que los DataFrames de recursos no estén vacíos antes de usarlos
    if insumos.empty: return None, "No se han definido insumos.", None, None, None
    if equipos.empty: return None, "No se han definido equipos.", None, None, None
    if personal.empty: return None, "No se ha definido personal.", None, None, None

    costo_insumos_por_producto = []
    costo_personal_por_producto = []

    for _, prod in productos.iterrows():
        costo_i, costo_p = 0, 0
        receta_prod = recetas[recetas['Producto'] == prod['Nombre']]
        
        for _, item_receta in receta_prod.iterrows():
            if item_receta['Tipo'] == 'Insumo':
                insumo_data = insumos[insumos['Nombre'] == item_receta['Recurso']]
                if not insumo_data.empty:
                    costo_insumo_unitario = insumo_data['Costo Unitario'].values[0]
                    costo_i += item_receta['Cantidad'] * costo_insumo_unitario
                else:
                    st.warning(f"Insumo '{item_receta['Recurso']}' en receta de '{prod['Nombre']}' no encontrado. Se ignorará.")
            elif item_receta['Tipo'] == 'Personal':
                personal_data = personal[personal['Rol'] == item_receta['Recurso']]
                if not personal_data.empty:
                    costo_hora_personal = personal_data['Costo por Hora'].values[0]
                    costo_p += item_receta['Cantidad'] * costo_hora_personal
                else:
                    st.warning(f"Personal '{item_receta['Recurso']}' en receta de '{prod['Nombre']}' no encontrado. Se ignorará.")
        
        costo_insumos_por_producto.append(costo_i)
        costo_personal_por_producto.append(costo_p)
    
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
    if not api_key or api_key == "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx": # Asumiendo un valor placeholder
        return "Por favor, introduce tu API Key de Hugging Face en la barra lateral para usar el análisis con IA."
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
            # Asegurarse de que el modelo Llama 3.1 no se cargue en modo chat si no es necesario,
            # aunque InferenceClient maneja esto bien con text_generation.
        )
        return response
    except Exception as e:
        return f"Error al contactar la API de Hugging Face: {e}. Asegúrate de que tu API Key sea válida y que el modelo 'meta-llama/Meta-Llama-3.1-8B-Instruct' esté disponible para tu cuenta."

# --- Interfaz de la App ---
st.title("💰 Optimizador de Rentabilidad Empresarial")

# --- Inicialización de Datos (ahora vacíos) ---
initial_data = get_empty_dataframes()
for key, df in initial_data.items():
    if key not in st.session_state:
        st.session_state[key] = df

# --- Limpieza de Datos en cada Rerun ---
# Se mantiene para limpiar recetas huérfanas si el usuario borra productos o recursos
clean_up_data()

# --- Barra Lateral y Navegación ---
st.sidebar.header("Navegación")
page = st.sidebar.radio("Ir a:", ["⚙️ 1. Configuración de Recursos", "📝 2. Definición de Procesos", "📈 3. Parámetros Financieros", "🚀 4. Optimización y Resultados", "🧠 5. Análisis con IA"])
st.sidebar.header("🔑 Configuración API")
hf_api_key = None
try:
    hf_api_key = st.secrets["HF_API_KEY"]
    st.sidebar.success("✅ API Key cargada desde Streamlit Secrets.")
except:
    st.sidebar.warning("API Key de Hugging Face no encontrada en Streamlit Secrets.")
    hf_api_key = st.sidebar.text_input("Ingresa tu Hugging Face API Key", type="password", help="Necesaria para el análisis con IA. Puedes obtenerla en huggingface.co/settings/tokens")
    if not hf_api_key:
        st.sidebar.info("Por favor, introduce tu API Key para usar la función de IA.")


# --- Contenido de las Páginas ---
if page == "⚙️ 1. Configuración de Recursos":
    st.header("1. Configuración de Recursos")
    st.subheader("A. Productos o Servicios")
    st.info("Añade, edita o elimina tus productos. Asegúrate de darles nombres únicos.")
    st.session_state.productos = st.data_editor(st.session_state.productos, num_rows="dynamic", key="productos_editor")
    
    st.subheader("B. Insumos / Materias Primas")
    st.info("Define tus insumos, su disponibilidad y costo unitario.")
    st.session_state.insumos = st.data_editor(st.session_state.insumos, num_rows="dynamic", key="insumos_editor")
    
    st.subheader("C. Equipos / Maquinaria")
    st.info("Registra tus equipos y sus horas disponibles para la producción.")
    st.session_state.equipos = st.data_editor(st.session_state.equipos, num_rows="dynamic", key="equipos_editor")
    
    st.subheader("D. Personal")
    st.info("Configura los roles de personal, su cantidad, horas de trabajo y costo por hora.")
    st.session_state.personal = st.data_editor(st.session_state.personal, num_rows="dynamic", key="personal_editor")

elif page == "📝 2. Definición de Procesos":
    st.header("2. Definición de Procesos (Recetas)")
    st.info("Define los recursos (insumos, equipos, personal) que cada producto necesita y en qué cantidad.")

    # Pre-calculamos las opciones para los selectbox individuales
    productos_validos = list(st.session_state.productos['Nombre'].unique())
    tipos_recurso_validos = ['Insumo', 'Equipo', 'Personal']

    st.subheader("Agregar Nueva Receta")
    if not productos_validos:
        st.warning("Por favor, define al menos un producto en la sección '1. Configuración de Recursos' antes de añadir recetas.")
    else:
        with st.form("add_recipe_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Si no hay productos, la opción por defecto es "---" y el selector está deshabilitado
                selected_producto = st.selectbox("Producto", 
                                                options=['---'] + productos_validos, 
                                                index=0, # Por defecto seleccionado "---"
                                                key="new_recipe_producto")
            with col2:
                selected_tipo = st.selectbox("Tipo de Recurso", options=tipos_recurso_validos, key="new_recipe_tipo")
            
            recursos_disponibles = []
            if selected_tipo == 'Insumo':
                recursos_disponibles = list(st.session_state.insumos['Nombre'].unique())
            elif selected_tipo == 'Equipo':
                recursos_disponibles = list(st.session_state.equipos['Nombre'].unique())
            elif selected_tipo == 'Personal':
                recursos_disponibles = list(st.session_state.personal['Rol'].unique())

            with col3:
                # Si no hay recursos disponibles para el tipo seleccionado, la opción por defecto es "---"
                selected_recurso = st.selectbox("Recurso Específico", 
                                                options=['---'] + recursos_disponibles, 
                                                index=0, # Por defecto seleccionado "---"
                                                key="new_recipe_recurso")
            with col4:
                new_cantidad = st.number_input("Cantidad", min_value=0.01, value=1.0, step=0.1, key="new_recipe_cantidad")
            
            submit_button = st.form_submit_button("Añadir Receta")
            
            if submit_button:
                if selected_producto == '---' or selected_recurso == '---':
                    st.error("Por favor, selecciona un Producto y un Recurso válidos.")
                elif new_cantidad <= 0:
                    st.error("La cantidad debe ser mayor que 0.")
                else:
                    # Comprobar si ya existe una receta idéntica
                    if not st.session_state.recetas[(st.session_state.recetas['Producto'] == selected_producto) &
                                                    (st.session_state.recetas['Tipo'] == selected_tipo) &
                                                    (st.session_state.recetas['Recurso'] == selected_recurso)].empty:
                        st.warning("Esta receta ya existe. Edítala en la tabla de abajo si quieres cambiar la cantidad.")
                    else:
                        new_row = pd.DataFrame([{'Producto': selected_producto, 'Tipo': selected_tipo, 'Recurso': selected_recurso, 'Cantidad': new_cantidad}])
                        st.session_state.recetas = pd.concat([st.session_state.recetas, new_row], ignore_index=True)
                        st.success(f"Receta para {selected_producto} usando {selected_recurso} añadida.")
        
    st.divider()

    st.subheader("Editar o Eliminar Recetas Existentes")
    st.info("Puedes editar directamente las cantidades o eliminar filas. Asegúrate de que los nombres de Producto, Tipo y Recurso sean exactos a los que definiste.")
    
    st.session_state.recetas = st.data_editor(
        st.session_state.recetas,
        num_rows="dynamic",
        key="editor_recetas_final", 
        column_config={
            "Producto": st.column_config.Column("Producto", help="Producto al que aplica esta receta"),
            "Tipo": st.column_config.Column("Tipo", help="Tipo de recurso (Insumo, Equipo, Personal)"),
            "Recurso": st.column_config.Column("Recurso", help="Nombre del insumo, equipo o rol de personal"),
            "Cantidad": st.column_config.NumberColumn("Cantidad", help="Cantidad necesaria por unidad de producto", min_value=0.01),
        }
    )

elif page == "📈 3. Parámetros Financieros":
    st.header("3. Parámetros Financieros y de Mercado")
    st.session_state.params['iibb'] = st.number_input("Tasa de Ingresos Brutos (%)", 0.0, 100.0, st.session_state.params.get('iibb', 3.5), 0.1, help="Impuesto sobre Ingresos Brutos aplicable a las ventas.")
    st.session_state.params['costo_capital'] = st.number_input("Costo de Capital / Financiero (%)", 0.0, 100.0, st.session_state.params.get('costo_capital', 8.0), 0.5, help="Costo asociado al capital invertido o financiamiento.")

elif page == "🚀 4. Optimización y Resultados":
    st.header("4. Optimización y Resultados")
    st.info("Haz clic en 'Ejecutar Optimización' para calcular el plan de producción que maximiza el beneficio.")
    if st.button("▶️ Ejecutar Optimización", type="primary"):
        if st.session_state.productos.empty:
            st.error("No hay productos definidos. Por favor, añádelos en la sección '1. Configuración de Recursos'.")
        elif st.session_state.recetas.empty:
            st.error("No hay recetas definidas. Por favor, añádelas en la sección '2. Definición de Procesos'.")
        else:
            with st.spinner("Calculando..."):
                res, msg, A, b, costs = optimizar_produccion(
                    st.session_state.productos, 
                    st.session_state.insumos, 
                    st.session_state.equipos, 
                    st.session_state.personal, 
                    st.session_state.recetas, 
                    st.session_state.params
                )
            if msg: 
                st.error(f"Error en la optimización: {msg}")
                # Limpiar resultados anteriores si la optimización falla
                if 'resultados_optimizacion' in st.session_state:
                    del st.session_state.resultados_optimizacion
                if 'produccion_optima' in st.session_state:
                    del st.session_state.produccion_optima
            else:
                st.success("¡Optimización completada!")
                st.session_state.resultados_optimizacion, st.session_state.A_ub, st.session_state.b_ub, st.session_state.costos_variables = res, A, b, costs
                # Asegurarse de que el DataFrame de producción óptima se construye correctamente
                if res and res.x is not None and len(res.x) == len(st.session_state.productos):
                    st.session_state.produccion_optima = pd.DataFrame({'Producto': st.session_state.productos['Nombre'], 'Cantidad a Producir': res.x})
                else:
                    st.error("Error al generar el plan de producción. Los resultados de la optimización pueden ser inválidos.")
                    if 'produccion_optima' in st.session_state: del st.session_state.produccion_optima

    if 'resultados_optimizacion' in st.session_state and st.session_state.resultados_optimizacion: # Verificar que haya resultados válidos
        res, costs = st.session_state.resultados_optimizacion, st.session_state.costos_variables
        
        beneficio_bruto = -res.fun
        
        # Validar la longitud de los arrays de costos antes de la multiplicación de matrices
        if res.x is None or len(res.x) != len(costs['insumos']) or len(res.x) != len(costs['personal']):
            st.error("Error: La longitud de los resultados de producción no coincide con los costos unitarios. Por favor, revisa tus recetas y la configuración de recursos.")
            beneficio_neto = 0 # Establecer a 0 para evitar más errores
            costo_financiero = 0
        else:
            costo_total_variable = np.dot(res.x, costs['insumos']) + np.dot(res.x, costs['personal'])
            tasa_capital = st.session_state.params.get('costo_capital', 0) / 100
            costo_financiero = costo_total_variable * tasa_capital 
            beneficio_neto = beneficio_bruto - costo_financiero
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Beneficio Bruto Óptimo", f"${beneficio_bruto:,.2f}")
        c2.metric("Costo Financiero", f"${costo_financiero:,.2f}", delta=f"-{st.session_state.params.get('costo_capital', 0)}%", delta_color="inverse")
        c3.metric("Beneficio Neto Final", f"${beneficio_neto:,.2f}")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Plan de Producción")
            if 'produccion_optima' in st.session_state and not st.session_state.produccion_optima.empty:
                df_prod = st.session_state.produccion_optima
                st.dataframe(df_prod[df_prod['Cantidad a Producir'] > 0.01].sort_values(by='Cantidad a Producir', ascending=False), use_container_width=True)
            else:
                st.info("No hay un plan de producción óptimo para mostrar.")
        with c2:
            st.subheader("Uso de Recursos")
            # Construir las etiquetas de las restricciones dinámicamente
            labels = []
            if not st.session_state.insumos.empty:
                for _, insumo in st.session_state.insumos.iterrows(): labels.append(f"Insumo: {insumo['Nombre']}")
            if not st.session_state.equipos.empty:
                for _, equipo in st.session_state.equipos.iterrows(): labels.append(f"Equipo: {equipo['Nombre']}")
            if not st.session_state.personal.empty:
                for _, p in st.session_state.personal.iterrows(): labels.append(f"Personal: {p['Rol']}")
            if not st.session_state.productos.empty:
                for _, prod in st.session_state.productos.iterrows(): labels.append(f"Demanda: {prod['Nombre']}")
            
            recursos_usados = st.session_state.A_ub @ res.x
            
            if len(labels) == len(recursos_usados) == len(st.session_state.b_ub):
                df_uso = pd.DataFrame({'Restricción': labels, 'Usado': recursos_usados, 'Disponible': st.session_state.b_ub})
                df_uso['Uso (%)'] = np.where(df_uso['Disponible'] > 0, (df_uso['Usado'] / df_uso['Disponible']) * 100, 0)
                st.dataframe(df_uso.sort_values(by='Uso (%)', ascending=False), use_container_width=True)
                st.session_state.uso_recursos = df_uso
            else:
                st.error("Error al mostrar el uso de recursos. La optimización pudo haber fallado parcialmente o los datos de las restricciones no coinciden con las etiquetas.")

elif page == "🧠 5. Análisis con IA":
    st.header("5. Análisis con IA y Contexto de Mercado")
    st.info("Utiliza Llama 3.1 para obtener insights sobre tus resultados de optimización y el contexto del mercado.")

    if 'resultados_optimizacion' not in st.session_state or not st.session_state.resultados_optimizacion: 
        st.warning("Ejecuta la optimización primero para generar un análisis con IA.")
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
        
        res = st.session_state.resultados_optimizacion
        costs = st.session_state.costos_variables
        
        beneficio_neto = 0 # Valor por defecto
        if res.x is not None and len(res.x) == len(costs['insumos']) and len(res.x) == len(costs['personal']):
            beneficio_bruto = -res.fun
            costo_total_variable = np.dot(res.x, costs['insumos']) + np.dot(res.x, costs['personal'])
            tasa_capital = st.session_state.params.get('costo_capital', 0) / 100
            costo_financiero = costo_total_variable * tasa_capital
            beneficio_neto = beneficio_bruto - costo_financiero
        else:
            st.error("No se pudo calcular el Beneficio Neto para la IA debido a inconsistencias en los resultados de optimización. Ejecuta la optimización de nuevo.")
        
        contexto_interno = f"Resultados de Optimización:\n- Beneficio Neto Final: ${beneficio_neto:,.2f}\n\nProducción Óptima:\n{st.session_state.produccion_optima.to_string()}\n\nUso de Recursos (Cuellos de Botella):\n{st.session_state.uso_recursos.to_string()}"
        
        contexto_completo = contexto_interno
        if market_context:
            contexto_completo += f"\n\nAnálisis de Mercado (del PDF):\n{market_context}"

        st.text_area("Contexto final enviado a la IA:", contexto_completo, height=300)
        pregunta = st.text_input("Haz tu pregunta:", "Basado en los datos de optimización y el contexto de mercado, ¿cuál debería ser mi principal foco estratégico?")
        
        if st.button("Obtener Insight con Llama 3.1", type="primary"):
            if not hf_api_key or hf_api_key == "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx":
                st.error("Por favor, configura tu API Key de Hugging Face en la barra lateral.")
            else:
                with st.spinner("Llama 3.1 está pensando..."):
                    respuesta = call_llama_api(hf_api_key, contexto_completo, pregunta)
                    st.success("Análisis recibido:")
                    st.markdown(respuesta)
