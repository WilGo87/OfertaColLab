import streamlit as st
from appsPag.main import *
from appsPag.data import *

selecNav = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Bienvenida", "Exploración de los datos", "Modelo ACP", "Modelo Clúster Jerárquico", "Modelo Clúster Kmeans", "Conclusiones"],
)

if selecNav == "Home":
    st.title("Presentación proyecto Demanda Laboral Colombia")
    st.markdown(
        """
    # **Métodos Estadísticos**
    >>**Facultad de Ingeniería y Ciencias Básicas.**

    >>**Universidad Central  2021 - I**

    
    ## Integrantes
    *  Diana Carolina Vargas Bornachera
    *  Yuliana Andrea Jaramillo Yarce  
    *  Wilmar José Gómez Soler

    >>*Profesor: Nelson Alirio Cruz*

    """
    )
elif selecNav == "Bienvenida":
    load_page_princ()
elif selecNav == "Exploración de los datos":
    expl_data_sec()
elif selecNav == "Modelo ACP":
    model_data_sec()
elif selecNav == "Modelo Clúster Jerárquico":
    model_agrup_jerarq(corte=5)
elif selecNav == "Modelo Clúster Kmeans":
    model_agrup_kmeans()
elif selecNav == "Conclusiones":
    load_Conclus()