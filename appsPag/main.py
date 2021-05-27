import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

header = st.beta_container()


def load_page_princ():
    st.title("Demanda Laboral en Colombia -- 2015/2020")
    st.markdown(
        """
        De acuerdo con la contingencia de salud que se presentó a nivel mundial, quisimos saber, cómo impactó a Colombia esta situación respecto a la oferta laboral para el año 2020, por tal razón, nos vimos motivados a consultar en las páginas adscritas al Ministerio del Trabajo que brindan información sobre las oportunidades de trabajo a nivel nacional, teniendo en cuenta variables como nivel de educación, salarios, experiencia laboral, sectores económicos y ocupación, distribuido por departamentos. De acuerdo con esta inquietud, nos apoyamos en la Unidad del Servicio Público de Empleo, la cual tiene como objetivo brindar información pública sobre ofertas laborales tanto en el sector público como privado. 
        Esta entidad ofrece datos abiertos de estudios e investigación por periodos y departamentos que pueden ser descargados para su uso. 

        [Fuente de información: Unidad del servicio de Empleo](https://www.serviciodeempleo.gov.co/estudios-e-investigacion/oferta-y-demanda-laboral/anexo-estadistico-de-demanda-laboral-vacantes)

        Este proyecto tiene como finalidad analizar las tendencias y características que más son buscadas por los prestadores de servicios de empleo durante el año 2020, teniendo en cuenta que este año, corresponde a un hito histórico a nivel mundial que afectó al sector productivo en diferentes ámbitos tanto negativa como positivamente:

        > 1. Educación
        > 2. Ocupación
        > 3. Salario
        > 4. Experiencia
        > 5. Sector

        """
    )


def load_Conclus():
    st.title("Conclusiones")
    st.markdown(
        """
        * De acuerdo con los resultados obtenidos por cada aspecto analizado, se identificó que las
        ofertas laborales para el año 2020, disminuyeron respecto al año 2019, lo que indica que si se
        vio impactado el sector laboral debido a la contingencia de salud que se vive a nivel mundial.
        
        * Se identificó que Casanare un comportamiento especial, por ser un departamento que proporcional
        a su poblacicón cuenta con un alto número de vacantes y el sector más relevante es
        la explotación de minas y canteras.

        * Los departamentos que más requieren vacantes con estudios de maestría corresponden Bogotá D.C, 
        Antioquia, Santander y Valle del Cauca.
        
        * El departamento de Archipiélago San Andrés presenta las mayores ofertas en Alojamiento y
        servicios de comida comparado con los demás departamentos.
        
        * Casanare es el departamento que más solicita meses de experiencia en sus ofertas (más de 60
         meses) laborales y la remuneración económica es asignada a convenir, no especifica los estudios
         económicos y sus ofertas dirigidas a la explotación de minas y canteras, la construcción
        y agricultura son las de mayor porcentaje comparado con los demás departamentos.

        """
    )
