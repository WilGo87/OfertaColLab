from numpy.core.numeric import correlate
import streamlit as st
from appsPag.modelOfertaV import *
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

import base64

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
    fcluster,
)  ### pintar dendrograma
from sklearn.cluster import AgglomerativeClustering  ## Calcular Agrupamiento jerarquico
import scipy.cluster.hierarchy as sch  ## DIstancias

import pickle

dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()


def expl_data_sec():

    # ----------------------------------------  Datos Originales -----------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    y_axis = st.sidebar.selectbox(
        "¿Qué nivel quiere seleccionar?",
        ["Ocupaciones", "Sectores", "Educación", "Experiencia", "Salarios"],
    )

    categoria = "Nivel educativo"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    if y_axis == "Ocupaciones":
        categoria = "Grupo Ocupacional"
    elif y_axis == "Sectores":
        categoria = "Sector empresarial Actividades"
    elif y_axis == "Educación":
        categoria = "Nivel educativo"
    elif y_axis == "Experiencia":
        categoria = "Experiencia laboral"
    elif y_axis == "Salarios":
        categoria = "Rangos salariales"

    st.header("Información detallada de la base de " + y_axis)

    oferV = OfertaLaboral(baseIni, y_axis, categoria, codMpio, codDpto, codPoblac)
    base = oferV.lecturaBase()
    base = oferV.cargaBaseT(base)

    baseCol = base.drop(
        ["AÑO", "DPNOM", "ÁREA GEOGRÁFICA"],
        axis=1,
    )

    st.dataframe(baseCol.head(5))
    st.write(f"Cantidad de filas y columnas", baseCol.shape)
    baseCol = base.drop(
        [
            "Código DIVIPOLA",
            "Departamento",
            "Población",
        ],
        axis=1,
    )

    y_categ = st.selectbox("¿Que categoria desea seleccionar?", baseCol.columns.values)

    baseAgrup = pd.DataFrame(base.groupby("Departamento")[y_categ].sum())

    st.markdown(get_table_download_link_csv(baseCol), unsafe_allow_html=True)

    baseAgrup = baseAgrup.rename_axis("Departamento").reset_index()
    baseAgrup = baseAgrup.sort_values([y_categ], ascending=False)

    fig = px.bar(baseAgrup, x="Departamento", y=y_categ)

    st.plotly_chart(fig)

    # ----------------------------------------  Union Bases -----------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    dict_base = {
        "Ocupaciones": "Grupo Ocupacional",
        "Sectores": "Sector empresarial Actividades",
        "Educación": "Nivel educativo",
        "Experiencia": "Experiencia laboral",
        "Salarios": "Rangos salariales",
    }

    for y_axis, categoria in dict_base.items():

        if y_axis == "Ocupaciones":

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )
            base_Ini = oferV.lecturaBase()

        else:

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )
            base = oferV.lecturaBase()

            base_Ini = pd.merge(
                base_Ini,
                base,
                on=[
                    "Código DIVIPOLA",
                    "Departamento",
                    "Población",
                    "AÑO",
                    "DPNOM",
                    "ÁREA GEOGRÁFICA",
                ],
            )

    base = oferV.cargaBaseT(base_Ini)

    baseCol = base.drop(
        [
            "AÑO",
            "DPNOM",
            "ÁREA GEOGRÁFICA",
        ],
        axis=1,
    )

    st.header("Información detallada de la base Variables Unificadas")

    st.dataframe(baseCol)
    st.write(f"Cantidad de filas y columnas", base.shape)

    baseCol = base.drop(
        ["AÑO", "DPNOM", "ÁREA GEOGRÁFICA", "Departamento"],
        axis=1,
    )

    st.write(baseCol.sum())


def model_data_sec():

    # ------------------------------------------- Datos Originales ------------------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    y_axis = st.sidebar.selectbox(
        "¿Qué nivel quiere seleccionar?",
        ["Ocupaciones", "Sectores", "Educación", "Experiencia", "Salarios"],
    )

    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    if y_axis == "Ocupaciones":
        categoria = "Grupo Ocupacional"
    elif y_axis == "Sectores":
        categoria = "Sector empresarial Actividades"
    elif y_axis == "Educación":
        categoria = "Nivel educativo"
    elif y_axis == "Experiencia":
        categoria = "Experiencia laboral"
    elif y_axis == "Salarios":
        categoria = "Rangos salariales"

    oferV = OfertaLaboral(baseIni, y_axis, categoria, codMpio, codDpto, codPoblac)
    base = oferV.lecturaBase()
    base = oferV.cargaBaseT(base)
    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCA()

    base = base[base["Código DIVIPOLA"] != "ND"]

    st.header("Análisis de Componentes datos originales " + y_axis)
    st.write(resulFin)
    fig = px.scatter(Pca_Tra, x="PC1", y="PC2", color=base["Departamento"])

    for i, feature in enumerate(features):

        fig.add_shape(type="line", x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )

    st.plotly_chart(fig)

    # ---------------------------------------------- Tasas Ocupacionales ---------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    oferV = OfertaLaboral(baseIni, y_axis, categoria, codMpio, codDpto, codPoblac)
    base = oferV.lecturaBase()
    base = oferV.cargaBaseT(base)

    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCATasas()

    base = base[base["Código DIVIPOLA"] != "ND"]

    st.header("Análisis de Componentes tasa poblacional " + y_axis)
    st.write(resulFin)
    fig = px.scatter(Pca_Tra, x="PC1", y="PC2", color=base["Departamento"])

    for i, feature in enumerate(features):

        fig.add_shape(type="line", x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )

    st.plotly_chart(fig)

    # ----------------------------------------  Union Bases -----------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    dict_base = {
        "Ocupaciones": "Grupo Ocupacional",
        "Sectores": "Sector empresarial Actividades",
        "Educación": "Nivel educativo",
        "Experiencia": "Experiencia laboral",
        "Salarios": "Rangos salariales",
    }

    for y_axis, categoria in dict_base.items():

        if y_axis == "Ocupaciones":

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )
            base_Ini = oferV.lecturaBase()

        else:

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )
            base = oferV.lecturaBase()

            base_Ini = pd.merge(
                base_Ini,
                base,
                on=[
                    "Código DIVIPOLA",
                    "Departamento",
                    "Población",
                    "AÑO",
                    "DPNOM",
                    "ÁREA GEOGRÁFICA",
                ],
            )

    base = oferV.cargaBaseT(base_Ini)
    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCATasas()

    base = base[base["Código DIVIPOLA"] != "ND"]

    st.header("Análisis de Componentes tasa poblacional - Variables Unificadas ")
    st.write(resulFin)
    fig = px.scatter(Pca_Tra, x="PC1", y="PC2", color=base["Departamento"])

    for i, feature in enumerate(features):

        fig.add_shape(type="line", x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )

    st.plotly_chart(fig)


def model_agrup_jerarq(corte=3):

    # ------------------------------------------- Datos Originales ------------------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    y_axis = st.sidebar.selectbox(
        "¿Qué nivel quiere seleccionar?",
        ["Ocupaciones", "Sectores", "Educación", "Experiencia", "Salarios"],
    )

    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    if y_axis == "Ocupaciones":
        categoria = "Grupo Ocupacional"
    elif y_axis == "Sectores":
        categoria = "Sector empresarial Actividades"
    elif y_axis == "Educación":
        categoria = "Nivel educativo"
    elif y_axis == "Experiencia":
        categoria = "Experiencia laboral"
    elif y_axis == "Salarios":
        categoria = "Rangos salariales"

    oferV = OfertaLaboral(baseIni, y_axis, categoria, codMpio, codDpto, codPoblac)
    base = oferV.lecturaBase()
    base = oferV.cargaBaseT(base)
    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCA()

    base = base.drop(
        [
            "ÁREA GEOGRÁFICA",
            "Población",
        ],
        axis=1,
    )

    st.header("Agrupamiento Jerárquico datos originales " + y_axis)

    st.set_option("deprecation.showPyplotGlobalUse", False)
    clusterJerarq = linkage(Pca_Tra, method="complete", metric="euclidean")

    clusters = fcluster(
        clusterJerarq, t=corte, criterion="distance"
    )  # t es la altura del corte del dendrograma
    base["Clustering Jerárquico"] = clusters

    st.subheader("Base con el Clustering Jerárquico")
    st.dataframe(base)

    st.markdown(get_table_download_link_csv(base), unsafe_allow_html=True)

    base["Clustering Jerárquico"] = clusters

    baseNew = base
    baseNew = baseNew.groupby(
        ["Departamento", "Clustering Jerárquico"], as_index=False
    )["DPNOM"].count()

    st.subheader(f"Cantidad de grupos con altura de corte {corte}")
    st.write(baseNew)

    fig = px.bar(baseNew, x="Departamento", y="Clustering Jerárquico")

    st.plotly_chart(fig)

    base = base.set_index("Departamento")

    st.header("Dendograma datos originales " + y_axis)
    plt.rcParams["figure.figsize"] = (20, 10)
    dendrograma = sch.dendrogram(clusterJerarq, labels=base.index)
    st.balloons()
    st.pyplot()

    # ------------------------------------------- Tasas Ocupacionales ------------------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    oferV = OfertaLaboral(baseIni, y_axis, categoria, codMpio, codDpto, codPoblac)
    base = oferV.lecturaBase()
    base = oferV.cargaBaseT(base)

    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCATasas()

    st.header("Agrupamiento Jerárquico tasa poblacional " + y_axis)

    st.set_option("deprecation.showPyplotGlobalUse", False)
    clusterJerarq = linkage(Pca_Tra, method="complete", metric="euclidean")

    clusters = fcluster(
        clusterJerarq, t=corte, criterion="distance"
    )  # t es la altura del corte del dendrograma
    base["Clustering Jerárquico"] = clusters

    st.subheader("Base con el Clustering Jerárquico")

    for item, feature in enumerate(features):
        print(feature)
        # if base[feature].dtype == "int64":
        base[feature] = base[feature] / base["Población"]

    st.dataframe(base)

    st.markdown(get_table_download_link_csv(base), unsafe_allow_html=True)

    base["Clustering Jerárquico"] = clusters
    baseNew = base
    baseNew = baseNew.groupby(
        ["Departamento", "Clustering Jerárquico"], as_index=False
    )["DPNOM"].count()

    st.subheader(f"Cantidad de grupos con altura de corte {corte}")
    st.write(baseNew)

    fig = px.bar(baseNew, x="Departamento", y="Clustering Jerárquico")

    st.plotly_chart(fig)

    base = base.set_index("Departamento")

    st.header("Dendograma tasa poblacional " + y_axis)
    plt.rcParams["figure.figsize"] = (20, 10)
    dendrograma = sch.dendrogram(clusterJerarq, labels=base.index)
    st.balloons()
    st.pyplot()

    # ----------------------------------------  Union Bases -----------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    dict_base = {
        "Ocupaciones": "Grupo Ocupacional",
        "Sectores": "Sector empresarial Actividades",
        "Educación": "Nivel educativo",
        "Experiencia": "Experiencia laboral",
        "Salarios": "Rangos salariales",
    }

    for y_axis, categoria in dict_base.items():

        if y_axis == "Ocupaciones":

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )
            base_Ini = oferV.lecturaBase()

        else:

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )

            base = oferV.lecturaBase()

            base_Ini = pd.merge(
                base_Ini,
                base,
                on=[
                    "Código DIVIPOLA",
                    "Departamento",
                    "Población",
                    "AÑO",
                    "DPNOM",
                    "ÁREA GEOGRÁFICA",
                ],
            )

    base = oferV.cargaBaseT(base_Ini)

    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCATasas()

    st.header("Agrupamiento Jerárquico tasa poblacional - Variables Unificadas")

    st.set_option("deprecation.showPyplotGlobalUse", False)
    clusterJerarq = linkage(Pca_Tra, method="complete", metric="euclidean")

    clusters = fcluster(
        clusterJerarq, t=corte, criterion="distance"
    )  # t es la altura del corte del dendrograma
    base["Clustering Jerárquico"] = clusters
    base["PoblaciónT"] = base["Población"]
    base["DepartamentoT"] = base["Departamento"]

    st.subheader("Base con el Clustering Jerárquico")

    features = base.columns.values
    for item, feature in enumerate(features):
        if base[feature].dtypes == "int64" or base[feature].dtypes == "float64":
            base[feature] = base[feature] / base["PoblaciónT"]
            print(base[feature])

    base = base.drop(["AÑO", "ÁREA GEOGRÁFICA", "Población", "DepartamentoT"], axis=1)

    st.dataframe(base)

    st.markdown(get_table_download_link_csv(base), unsafe_allow_html=True)

    base["Clustering Jerárquico"] = clusters

    baseNew = base
    baseNew = baseNew.groupby(
        ["Departamento", "Clustering Jerárquico"], as_index=False
    )["DPNOM"].count()

    st.subheader(f"Cantidad de grupos con altura de corte {corte}")
    st.write(baseNew)

    fig = px.bar(baseNew, x="Departamento", y="Clustering Jerárquico")

    st.plotly_chart(fig)

    base = base.set_index("Departamento")

    st.header("Dendograma tasa poblacional - Variables Unificadas")
    plt.rcParams["figure.figsize"] = (20, 10)
    dendrograma = sch.dendrogram(clusterJerarq, labels=base.index)
    st.balloons()
    st.pyplot()


def model_agrup_kmeans():

    # ------------------------------------------- Datos Originales ------------------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    y_axis = st.sidebar.selectbox(
        "¿Qué nivel quiere seleccionar?",
        ["Ocupaciones", "Sectores", "Educación", "Experiencia", "Salarios"],
    )

    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    if y_axis == "Ocupaciones":
        categoria = "Grupo Ocupacional"
    elif y_axis == "Sectores":
        categoria = "Sector empresarial Actividades"
    elif y_axis == "Educación":
        categoria = "Nivel educativo"
    elif y_axis == "Experiencia":
        categoria = "Experiencia laboral"
    elif y_axis == "Salarios":
        categoria = "Rangos salariales"

    oferV = OfertaLaboral(baseIni, y_axis, categoria, codMpio, codDpto, codPoblac)
    base = oferV.lecturaBase()
    base = oferV.cargaBaseT(base)
    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCA()

    st.header("K means datos originales " + y_axis)
    st.write(resulFin)

    within = []  ## Elbow Graph (codo), se tiende a elegir muchos grupos
    for k in range(1, 10):
        kmeanModel = KMeans(n_clusters=k).fit(cuanti_pca)
        within.append(kmeanModel.inertia_)
    fig = px.line(x=list(range(1, 10)), y=within, title="Codo de Jambú")

    st.plotly_chart(fig)

    tamanho, promed, matriz, Pca_Tra, etiquetas = oferV.kmeans(cuanti_pca, Pca_Tra)
    st.write("Tamaño de los grupos")
    st.write(tamanho)
    st.write("Promedio de los grupos")
    st.write(promed)
    st.write("Clasificación de los grupos")
    st.write(matriz)
    st.markdown(get_table_download_link_csv(matriz), unsafe_allow_html=True)

    colores = ["green", "blue", "red"]

    color_cluster = [colores[etiquetas[item]] for item in range(len(etiquetas))]

    st.subheader("Gráfica de clústers")

    fig = px.scatter(Pca_Tra, x="PC1", y="PC2", color=color_cluster)
    fig.add_trace(
        go.Scatter(
            x=promed["PC1"],
            y=promed["PC2"],
            text=["X_Centroide", "X_Centroide", "X_Centroide"],
            mode="text",
        )
    )

    for i, feature in enumerate(features):

        fig.add_shape(type="line", x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    st.plotly_chart(fig)

    st.balloons()

    # ------------------------------------------- Tasas Ocupacionales ------------------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    oferV = OfertaLaboral(baseIni, y_axis, categoria, codMpio, codDpto, codPoblac)
    base = oferV.lecturaBase()
    base = oferV.cargaBaseT(base)

    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCATasas()

    st.header("K means tasa poblacional " + y_axis)
    st.write(resulFin)

    within = []  ## Elbow Graph (codo), se tiende a elegir muchos grupos
    for k in range(1, 10):
        kmeanModel = KMeans(n_clusters=k).fit(cuanti_pca)
        within.append(kmeanModel.inertia_)
    fig = px.line(x=list(range(1, 10)), y=within, title="Codo de Jambú")
    st.plotly_chart(fig)

    tamanho, promed, matriz, Pca_Tra, etiquetas = oferV.kmeans(cuanti_pca, Pca_Tra)
    st.write("Tamaño de los grupos")
    st.write(tamanho)
    st.write("Promedio de los grupos")
    st.write(promed)
    st.write("Clasificación de los grupos")

    caract = matriz.columns.values
    for item, feature in enumerate(caract):

        if matriz[feature].dtypes == "int64":
            matriz[feature] = matriz[feature] / matriz["Población"]

    st.write(matriz)
    st.markdown(get_table_download_link_csv(matriz), unsafe_allow_html=True)

    st.subheader("Gráfica de clústers")

    colores = ["green", "blue", "red"]

    color_cluster = [colores[etiquetas[item]] for item in range(len(etiquetas))]

    fig = px.scatter(Pca_Tra, x="PC1", y="PC2", color=color_cluster)
    fig.add_trace(
        go.Scatter(
            x=promed["PC1"],
            y=promed["PC2"],
            text=["X_Centroide", "X_Centroide", "X_Centroide"],
            mode="text",
        )
    )
    for i, feature in enumerate(features):

        fig.add_shape(type="line", x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    st.plotly_chart(fig)

    st.balloons()

    # ----------------------------------------  Union Bases -----------------------------------------

    codDpto = "CodigoDpto.xlsx"
    codMpio = "CodigoMpio.xlsx"
    baseIni = "OfertaLaboral.xlsx"
    codPoblac = "OfertaLaboral_Poblac.xlsx"

    dict_base = {
        "Ocupaciones": "Grupo Ocupacional",
        "Sectores": "Sector empresarial Actividades",
        "Educación": "Nivel educativo",
        "Experiencia": "Experiencia laboral",
        "Salarios": "Rangos salariales",
    }

    for y_axis, categoria in dict_base.items():

        if y_axis == "Ocupaciones":

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )
            base_Ini = oferV.lecturaBase()

        else:

            oferV = OfertaLaboral(
                baseIni, y_axis, categoria, codMpio, codDpto, codPoblac
            )
            base = oferV.lecturaBase()

            base_Ini = pd.merge(
                base_Ini,
                base,
                on=[
                    "Código DIVIPOLA",
                    "Departamento",
                    "Población",
                    "AÑO",
                    "DPNOM",
                    "ÁREA GEOGRÁFICA",
                ],
            )

    base = oferV.cargaBaseT(base_Ini)

    (
        Pca_Tra,
        ajustPCA,
        expl,
        resulFin,
        features,
        loadings,
        cuanti_pca,
    ) = oferV.calculoPCATasas()

    st.header("K means tasa poblacional - Variables Unificadas")

    st.write(resulFin)

    within = []  ## Elbow Graph (codo), se tiende a elegir muchos grupos
    for k in range(1, 10):
        kmeanModel = KMeans(n_clusters=k).fit(cuanti_pca)
        within.append(kmeanModel.inertia_)
    fig = px.line(x=list(range(1, 10)), y=within, title="Codo de Jambú")
    st.plotly_chart(fig)

    tamanho, promed, matriz, Pca_Tra, etiquetas = oferV.kmeans(cuanti_pca, Pca_Tra)
    st.write("Tamaño de los grupos")
    st.write(tamanho)
    st.write("Promedio de los grupos")
    st.write(promed)
    st.write("Clasificación de los grupos")

    matriz["PoblaciónT"] = matriz["Población"]
    matriz["DepartamentoT"] = matriz["Departamento"]

    charact = matriz.columns.values
    for item, feature in enumerate(charact):
        if matriz[feature].dtypes == "int64" or matriz[feature].dtypes == "float64":
            matriz[feature] = matriz[feature] / matriz["PoblaciónT"]

    matriz = matriz.drop(
        ["AÑO", "ÁREA GEOGRÁFICA", "Población", "DepartamentoT"], axis=1
    )

    st.dataframe(matriz)
    st.markdown(get_table_download_link_csv(matriz), unsafe_allow_html=True)

    colores = ["green", "blue", "red"]

    color_cluster = [colores[etiquetas[item]] for item in range(len(etiquetas))]

    st.subheader("Gráfica de clústers")

    fig = px.scatter(Pca_Tra, x="PC1", y="PC2", color=color_cluster)
    fig.add_trace(
        go.Scatter(
            x=promed["PC1"],
            y=promed["PC2"],
            text=["X_Centroide", "X_Centroide", "X_Centroide"],
            mode="text",
        )
    )
    for i, feature in enumerate(features):

        fig.add_shape(type="line", x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    st.plotly_chart(fig)

    st.balloons()


def get_table_download_link_csv(df):
    # csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    # b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
    return href
