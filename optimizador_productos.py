import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from huggingface_hub import InferenceClient

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Optimizador de Rentabilidad Empresarial",
    page_icon="💰",
    layout="wide"
)

# --- Funciones Auxiliares ---

def optimizar_produccion(productos, insumos, equipos, personal, recetas, params):
    """
    Función principal para ejecutar la optimización con Programación Lineal.
    """
    num_productos = len(productos)
    if num_productos == 0:
        return None, "No se han definido productos para optimizar.", None, None, None

    # 1. Definir la función objetivo (Maximizar Beneficio)
    costo_insumos_por_producto = []
    costo_personal_por_producto = []
    
    for i, prod in productos.iterrows():
        costo_i = 0
        receta_prod = recetas[recetas['Producto'] == prod['Nombre']]
        for j, item_receta in receta_prod.iterrows():
            if item_receta['Tipo'] == 'Insumo':
                costo_insumo_unitario = insumos[insumos['Nombre'] == item_receta['Recurso']]['Costo Unitario'].values[0]
                costo_i += item_receta['Cantidad'] * costo_insumo_unitario
        costo_insumos_por_producto.append(costo_i)

        costo_p = 0
        for j, item_receta in receta_prod.iterrows():
            if item_receta['Tipo'] == 'Personal':
                costo_hora_personal = personal[personal['Rol'] == item_receta['Recurso']]['Costo por Hora'].values[0]
                costo_p += item_receta['Cantidad'] * costo_hora_personal
        costo_personal_por_producto.append(costo_p)

    precio_venta_neto = productos['Precio de Venta'].values * (1 - params['iibb'] / 100)
    beneficio_unitario = precio_venta_neto - np.array(costo_insumos_por_producto) - np.array(costo_personal_por_producto)
    c = -beneficio_unitario

    # 2. Definir las restricciones (A_ub * x <= b_ub)
    constraints_A = []
    constraints_b = []
    
    for i, insumo in insumos.iterrows():
        constraint_row = np.zeros(num_productos)
        for j, prod in productos.iterrows():
            cantidad_necesaria = recetas[(recetas['Producto'] == prod['Nombre']) & (recetas['Recurso'] == insumo['Nombre']) & (recetas['Tipo'] == 'Insumo')]['Cantidad'].sum()
            constraint_row[j] = cantidad_necesaria
        constraints_A.append(constraint_row)
        constraints_b.append(insumo['Cantidad Disponible'])

    for i, equipo in equipos.iterrows():
        constraint_row = np.zeros(num_productos)
        for j, prod in productos.iterrows():
            tiempo_necesario = recetas[(recetas['Producto'] == prod['Nombre']) & (recetas['Recurso'] == equipo['Nombre']) & (recetas['Tipo'] == 'Equipo')]['Cantidad'].sum()
            constraint_row[j] = tiempo_necesario
        constraints_A.append(constraint_row)
        constraints_b.append(equipo['Horas Disponibles'])

    for i, p in personal.iterrows():
        constraint_row = np.zeros(num_productos)
        for j, prod in productos.iterrows():
            tiempo_necesario = recetas[(recetas['Producto'] == prod['Nombre']) & (recetas['Recurso'] == p['Rol']) & (recetas['Tipo'] == 'Personal')]['Cantidad'].sum()
            constraint_row[j] = tiempo_necesario
        constraints_A.append(constraint_row)
        constraints_b.append(p['Cantidad de Empleados'] * p['Horas por Empleado'])
        
    for i, prod in productos.iterrows():
        constraint_row = np.zeros(num_productos)
        constraint_row[i] = 1
        constraints_A.append(constraint_row)
        constraints_b.append(prod['Demanda Máxima'])

    A_ub = np.array(constraints_A)
    b_ub = np.array(constraints_b)

    bounds = [(0, None) for _ in range(num_productos)]
    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    costos_variables_unitarios = {
        'insumos': np.array(costo_insumos_por_producto),
        'personal': np.array(costo_personal_por_producto)
    }

    if resultado.success:
        return resultado, None, A_ub, b_ub, costos_variables_unitarios
    else:
        return None, resultado.message, None, None, None

def call_huggingface_rag(api_key, context, question):
    """
    Llama a un modelo de Hugging Face para obtener insights.
    ### FIX DEFINITIVO ###: Se cambia a un modelo más estándar para text-generation.
    """
    if not api_key:
        return "Por favor, introduce tu API Key de Hugging Face en la barra lateral."
    
    try:
        client = InferenceClient(token=api_key)
        prompt = f"""
        **Contexto:**
        {context}

        **Pregunta:**
        {question}

        **Respuesta:**
        """
        response = client.text_generation(
            # Este modelo es muy fiable para la tarea de text-generation / instruction.
            model="google/flan-t5-xxl", # <--- CAMBIO FINAL Y ROBUSTO AQUÍ
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.8 
        )
        return response
    except Exception as e:
        return f"Error al contactar la API de Hugging Face: {e}"

# --- Interfaz de la App (El resto del código no cambia y está correcto) ---

st.title("💰 Optimizador de Rentabilidad Empresarial")
st.markdown("Una herramienta para maximizar tus beneficios encontrando el mix de producción ideal.")

# Barra Lateral
st.sidebar.header("🔑 Configuración API (Opcional)")
hf_api_key = st.sidebar.text_input("Hugging Face API Key", type="password", help="Necesaria para la función 'Análisis con IA'")
st.sidebar.header("Navegación")
page = st.sidebar.radio("Ir a:", ["⚙️ 1. Configuración de Recursos", "📝 2. Definición de Procesos", "📈 3. Parámetros Financieros", "🚀 4. Optimización y Resultados", "🧠 5. Análisis con IA"])

# Inicialización de Datos
if 'productos' not in st.session_state:
    st.session_state.productos = pd.DataFrame({'Nombre': ['Producto A', 'Producto B'], 'Demanda Máxima': [100, 150], 'Precio de Venta': [50.0, 75.0]})
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

# Contenido de las Páginas
if page == "⚙️ 1. Configuración de Recursos":
    st.header("1. Configuración de Recursos")
    st.markdown("Define los elementos básicos de tu operación. Puedes agregar, editar o eliminar filas.")
    st.subheader("A. Productos o Servicios")
    st.session_state.productos = st.data_editor(st.session_state.productos, num_rows="dynamic", key="editor_productos")
    st.subheader("B. Insumos / Materias Primas")
    st.session_state.insumos = st.data_editor(st.session_state.insumos, num_rows="dynamic", key="editor_insumos")
    st.subheader("C. Equipos / Maquinaria")
    st.session_state.equipos = st.data_editor(st.session_state.equipos, num_rows="dynamic", key="editor_equipos")
    st.subheader("D. Personal")
    st.session_state.personal = st.data_editor(st.session_state.personal, num_rows="dynamic", key="editor_personal")

elif page == "📝 2. Definición de Procesos":
    st.header("2. Definición de Procesos (Recetas)")
    st.markdown("""
    Aquí conectas tus recursos para definir cómo se crea cada producto.
    - **Producto:** Elige el producto que estás definiendo.
    - **Tipo:** Indica si el recurso es un Insumo, Equipo o Personal.
    - **Recurso:** Elige el recurso específico de la lista que definiste antes.
    - **Cantidad:**
        - Para **Insumos:** Cantidad de unidades del insumo por cada unidad de producto.
        - Para **Equipos:** Horas de uso de la máquina por cada unidad de producto.
        - Para **Personal:** Horas de trabajo de ese rol por cada unidad de producto.
    """)
    st.session_state.recetas = st.data_editor(st.session_state.recetas, num_rows="dynamic", key="editor_recetas",
        column_config={
            "Producto": st.column_config.SelectboxColumn("Producto", options=st.session_state.productos['Nombre'].unique(), required=True),
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=['Insumo', 'Equipo', 'Personal'], required=True),
            "Recurso": st.column_config.SelectboxColumn("Recurso", options=pd.concat([st.session_state.insumos['Nombre'], st.session_state.equipos['Nombre'], st.session_state.personal['Rol']]).unique(), required=True),
        }
    )

elif page == "📈 3. Parámetros Financieros":
    st.header("3. Parámetros Financieros y de Mercado")
    st.markdown("Define impuestos y otros costos que afectan la rentabilidad final.")
    st.session_state.params['iibb'] = st.number_input("Tasa de Ingresos Brutos (%)", min_value=0.0, max_value=100.0, value=st.session_state.params['iibb'], step=0.1, format="%.2f")
    st.session_state.params['costo_capital'] = st.number_input(
        "Costo de Capital / Financiero (%)", 
        min_value=0.0, max_value=100.0, 
        value=st.session_state.params.get('costo_capital', 8.0), 
        step=0.5, format="%.2f",
        help="Porcentaje aplicado sobre los costos variables totales (insumos + personal) para representar el costo de oportunidad del capital invertido."
    )

elif page == "🚀 4. Optimización y Resultados":
    st.header("4. Optimización y Resultados")
    if st.button("▶️ Ejecutar Optimización", type="primary"):
        with st.spinner("Calculando el mix de producción óptimo..."):
            resultado, mensaje_error, A_ub, b_ub, costos_variables = optimizar_produccion(
                st.session_state.productos, st.session_state.insumos, st.session_state.equipos,
                st.session_state.personal, st.session_state.recetas, st.session_state.params
            )
        if mensaje_error:
            st.error(f"Error en la optimización: {mensaje_error}")
        else:
            st.success("¡Optimización completada con éxito!")
            st.session_state.resultados_optimizacion = resultado
            st.session_state.A_ub = A_ub
            st.session_state.b_ub = b_ub
            st.session_state.costos_variables = costos_variables
            st.session_state.produccion_optima = pd.DataFrame({
                'Producto': st.session_state.productos['Nombre'],
                'Cantidad a Producir': resultado.x
            })

    if 'resultados_optimizacion' in st.session_state:
        st.subheader("Resultados Financieros")
        resultado = st.session_state.resultados_optimizacion
        costos_variables = st.session_state.costos_variables
        beneficio_bruto_optimo = -resultado.fun
        costo_total_insumos = np.dot(resultado.x, costos_variables['insumos'])
        costo_total_personal = np.dot(resultado.x, costos_variables['personal'])
        costo_variable_total = costo_total_insumos + costo_total_personal
        tasa_costo_capital = st.session_state.params.get('costo_capital', 0) / 100
        costo_financiero = costo_variable_total * tasa_costo_capital
        beneficio_neto_final = beneficio_bruto_optimo - costo_financiero

        col1, col2, col3 = st.columns(3)
        col1.metric("Beneficio Bruto Óptimo", f"${beneficio_bruto_optimo:,.2f}", help="Ingresos netos de IIBB menos costos variables directos (insumos, personal).")
        col2.metric("Costo Financiero", f"${costo_financiero:,.2f}", delta=f"(-{st.session_state.params.get('costo_capital', 0)}%)", delta_color="inverse", help="Costo de capital aplicado sobre el total de costos variables.")
        col3.metric("Beneficio Neto Final", f"${beneficio_neto_final:,.2f}", help="Beneficio bruto menos el costo financiero.")
        st.divider()

        st.subheader("Análisis de Producción y Recursos")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan de Producción Sugerido")
            produccion_df = st.session_state.produccion_optima
            st.dataframe(produccion_df[produccion_df['Cantidad a Producir'] > 0.01].style.format({'Cantidad a Producir': '{:,.2f}'}), use_container_width=True)
            st.bar_chart(produccion_df, x='Producto', y='Cantidad a Producir')

        with col2:
            st.subheader("Uso de Recursos (Restricciones)")
            A_ub = st.session_state.A_ub
            b_ub = st.session_state.b_ub
            recursos_usados = A_ub @ resultado.x
            
            constraint_labels = []
            for i, insumo in st.session_state.insumos.iterrows(): constraint_labels.append(f"Insumo: {insumo['Nombre']}")
            for i, equipo in st.session_state.equipos.iterrows(): constraint_labels.append(f"Equipo: {equipo['Nombre']}")
            for i, p in st.session_state.personal.iterrows(): constraint_labels.append(f"Personal: {p['Rol']}")
            for i, prod in st.session_state.productos.iterrows(): constraint_labels.append(f"Demanda: {prod['Nombre']}")

            uso_df = pd.DataFrame({
                'Restricción': constraint_labels,
                'Recurso Usado': recursos_usados,
                'Recurso Disponible': b_ub
            })
            uso_df['Porcentaje de Uso'] = np.where(uso_df['Recurso Disponible'] > 0, (uso_df['Recurso Usado'] / uso_df['Recurso Disponible']) * 100, 0)
            
            st.dataframe(uso_df.style.format({
                'Recurso Usado': '{:,.2f}',
                'Recurso Disponible': '{:,.2f}',
                'Porcentaje de Uso': '{:,.1f}%'
            }), use_container_width=True)
            
            st.warning("💡 Los recursos con un uso cercano al 100% son tus **cuellos de botella**. Mejorar su disponibilidad podría aumentar significativamente tu beneficio.")
            st.session_state.uso_recursos = uso_df

elif page == "🧠 5. Análisis con IA":
    st.header("5. Análisis con IA (RAG)")
    st.markdown("Haz preguntas en lenguaje natural sobre los resultados de la optimización.")
    if 'resultados_optimizacion' not in st.session_state:
        st.warning("Primero debes ejecutar la optimización en la pestaña '🚀 4. Optimización y Resultados'.")
    else:
        produccion_df = st.session_state.produccion_optima
        uso_df = st.session_state.uso_recursos
        beneficio_bruto = -st.session_state.resultados_optimizacion.fun
        costos_variables = st.session_state.costos_variables
        resultado = st.session_state.resultados_optimizacion
        costo_variable_total = np.dot(resultado.x, costos_variables['insumos']) + np.dot(resultado.x, costos_variables['personal'])
        tasa_costo_capital = st.session_state.params.get('costo_capital', 0) / 100
        costo_financiero = costo_variable_total * tasa_costo_capital
        beneficio_neto = beneficio_bruto - costo_financiero

        contexto_str = f"""
        Resultados de la Optimización de Rentabilidad:
        - Beneficio Bruto Máximo: ${beneficio_bruto:,.2f}
        - Costo Financiero ({st.session_state.params.get('costo_capital', 0)}%): -${costo_financiero:,.2f}
        - Beneficio Neto Final: ${beneficio_neto:,.2f}
        
        Plan de Producción Óptimo:
        {produccion_df.to_string(index=False)}

        Uso de Recursos y Cuellos de Botella:
        {uso_df.to_string(index=False)}

        Nota: Un recurso con Porcentaje de Uso cercano al 100% es un cuello de botella.
        """
        st.text_area("Contexto enviado a la IA:", contexto_str, height=300)
        pregunta_usuario = st.text_input("Haz tu pregunta aquí:", "Cual es mi principal cuello de botella y que producto lo consume mas?")
        if st.button("Obtener Insight", type="primary"):
            if not hf_api_key:
                st.error("Por favor, introduce tu API Key de Hugging Face en la barra lateral.")
            else:
                with st.spinner("Pensando..."):
                    respuesta = call_huggingface_rag(hf_api_key, contexto_str, pregunta_usuario)
                    st.success("Análisis recibido:")
                    st.markdown(respuesta)
