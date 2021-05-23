import streamlit as st
from appsPag.modelOfertaV import *
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

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

    st.dataframe(base.head(5))
    st.write(f"Cantidad de filas y columnas", base.shape)
    baseCol = base.drop(
        [
            "Código DIVIPOLA",
            "mes",
            "Departamento",
            "AÑO",
            "DPNOM",
            "ÁREA GEOGRÁFICA",
            "Población",
        ],
        axis=1,
    )

    y_categ = st.selectbox("¿Que categoria desea seleccionar?", baseCol.columns.values)

    baseAgrup = pd.DataFrame(base.groupby("Departamento")[y_categ].sum())
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
                    "mes",
                    "Departamento",
                    "Población",
                    "AÑO",
                    "DPNOM",
                    "ÁREA GEOGRÁFICA",
                ],
            )

    base = oferV.cargaBaseT(base_Ini)

    st.dataframe(base.head(5))
    st.write(f"Cantidad de filas y columnas", base.shape)
    baseCol = base.drop(
        [
            "Código DIVIPOLA",
            "mes",
            "Departamento",
            "AÑO",
            "DPNOM",
            "ÁREA GEOGRÁFICA",
            "Población",
        ],
        axis=1,
    )

    st.header("Información detallada de la base Variables Unificadas")

    st.dataframe(baseCol)
    # baseAgrup = pd.DataFrame(base.groupby("Departamento")[y_categ].sum())
    # baseAgrup = baseAgrup.rename_axis("Departamento").reset_index()
    # baseAgrup = baseAgrup.sort_values([y_categ], ascending=False)


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

    st.header("Información detallada de la Union de las bases " + y_axis)

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
                    "mes",
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


def model_agrup_jerarq():

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

    base = base.drop(
        [
            "mes",
            "DPNOM",
            "ÁREA GEOGRÁFICA",
            "Población",
        ],
        axis=1,
    )

    st.header("Agrupamiento Jerárquico " + y_axis)

    st.set_option("deprecation.showPyplotGlobalUse", False)
    clusterJerarq = linkage(Pca_Tra, method="ward", metric="euclidean")

    clusters = fcluster(
        clusterJerarq, t=2, criterion="distance"
    )  # t es la altura del corte del dendrograma
    base["Clustering Jerárquico"] = clusters

    st.subheader("Base con el Clustering Jerárquico")
    st.dataframe(base)

    base["Clustering Jerárquico"] = clusters

    baseNew = base
    baseNew = baseNew.groupby(
        ["Departamento", "Clustering Jerárquico"], as_index=False
    )["AÑO"].count()

    st.subheader("Cantidad de grupos con altura de corte 1")
    st.write(baseNew)

    fig = px.bar(baseNew, x="Departamento", y="Clustering Jerárquico")

    st.plotly_chart(fig)

    st.header("Dendograma " + y_axis)
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

    base = base[base["Código DIVIPOLA"] != "ND"]

    base = base.drop(
        [
            "mes",
            "DPNOM",
            "ÁREA GEOGRÁFICA",
            "Población",
        ],
        axis=1,
    )

    st.header("Agrupamiento Jerárquico " + y_axis)

    st.set_option("deprecation.showPyplotGlobalUse", False)
    clusterJerarq = linkage(Pca_Tra, method="ward", metric="euclidean")

    clusters = fcluster(
        clusterJerarq, t=2, criterion="distance"
    )  # t es la altura del corte del dendrograma
    base["Clustering Jerárquico"] = clusters

    st.subheader("Base con el Clustering Jerárquico")
    st.dataframe(base)

    base["Clustering Jerárquico"] = clusters

    baseNew = base
    baseNew = baseNew.groupby(
        ["Departamento", "Clustering Jerárquico"], as_index=False
    )["AÑO"].count()

    st.subheader("Cantidad de grupos con altura de corte 1")
    st.write(baseNew)

    fig = px.bar(baseNew, x="Departamento", y="Clustering Jerárquico")

    st.plotly_chart(fig)

    st.header("Dendograma " + y_axis)
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
                    "mes",
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

    base = base.drop(
        [
            "mes",
            "DPNOM",
            "ÁREA GEOGRÁFICA",
            "Población",
        ],
        axis=1,
    )

    st.header("Agrupamiento Jerárquico " + y_axis)

    st.set_option("deprecation.showPyplotGlobalUse", False)
    clusterJerarq = linkage(Pca_Tra, method="ward", metric="euclidean")

    clusters = fcluster(
        clusterJerarq, t=2, criterion="distance"
    )  # t es la altura del corte del dendrograma
    base["Clustering Jerárquico"] = clusters

    st.subheader("Base con el Clustering Jerárquico")
    st.dataframe(base)

    base["Clustering Jerárquico"] = clusters

    baseNew = base
    baseNew = baseNew.groupby(
        ["Departamento", "Clustering Jerárquico"], as_index=False
    )["AÑO"].count()

    st.subheader("Cantidad de grupos con altura de corte 1")
    st.write(baseNew)

    fig = px.bar(baseNew, x="Departamento", y="Clustering Jerárquico")

    st.plotly_chart(fig)

    st.header("Dendograma " + y_axis)
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

    base = base[base["Código DIVIPOLA"] != "ND"]

    st.header("K means datos originales " + y_axis)

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

    colores = ["green", "blue", "red"]

    color_cluster = [colores[etiquetas[item]] for item in range(len(etiquetas))]

    st.subheader("Gráfica de clústers")
    st.write(resulFin)

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

    st.header("K means tasas de población " + y_axis)
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

    colores = ["green", "blue", "red"]

    color_cluster = [colores[etiquetas[item]] for item in range(len(etiquetas))]

    st.subheader("Gráfica de clústers")
    st.write(resulFin)
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
                    "mes",
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
    st.write(matriz)

    colores = ["green", "blue", "red"]

    color_cluster = [colores[etiquetas[item]] for item in range(len(etiquetas))]

    st.subheader("Gráfica de clústers")
    st.write(resulFin)
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
