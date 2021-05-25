#%%

import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy import cluster
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import kruskal

from sklearn import preprocessing


# %%
class OfertaLaboral:
    def __init__(
        self, baseIni="", nomHoja="", nomCol="", codMpio="", codDpto="", codPoblac=""
    ):
        self.base = pd.read_excel(open(baseIni, "rb"), sheet_name=nomHoja, index_col=0)
        self.codMpio = pd.read_excel(open(codMpio, "rb"))
        self.codDpto = pd.read_excel(open(codDpto, "rb"))
        if codPoblac == "":
            self.codPoblac = pd.DataFrame()
        else:
            self.codPoblac = pd.read_excel(open(codPoblac, "rb"))
        self.baseOf = None
        self.dataB = None
        self.cuantiVar = None
        self.corre = None
        self.val = None
        self.vec = None
        self.covarianza = None
        self.nomCol = nomCol
        self.ConsolB = pd.DataFrame()
        self.ConsolP = pd.DataFrame()
        self.cont = 0

    # %%
    def lecturaBase(self):
        for valLoop in range(len(self.base.columns) - 1, 0, -2):
            mes = self.base.columns.values[valLoop]
            if mes == self.nomCol:
                break

            ConsolParc = pd.DataFrame()

            ConsolParc = pd.pivot_table(
                self.base,
                values="TOTAL",
                index="Código DIVIPOLA",
                columns=[self.nomCol],
                aggfunc=np.sum,
            )

            self.ConsolB.reset_index

            self.ConsolB = ConsolParc

            if self.codPoblac.size != 0:
                self.ConsolB = pd.merge(
                    self.ConsolB,
                    self.codPoblac,
                    how="left",
                    left_on=["Código DIVIPOLA"],
                    right_on=["Código DIVIPOLA"],
                )

            self.ConsolB = pd.merge(
                self.ConsolB,
                self.codDpto,
                how="left",
                left_on=["Código DIVIPOLA"],
                right_on=["Código DIVIPOLA"],
            )

        return self.ConsolB

    # %%

    def cargaBaseT(self, baseUb):
        self.baseOf = baseUb
        return self.baseOf

    # %%
    # Se realiza el cálculo del PCA
    def calculoPCA(self):
        # Seleccionamos las cuantitativas
        ## Porque PCA solo trabaja con cuantitaivas

        self.baseOf = self.baseOf[self.baseOf["Código DIVIPOLA"] != "ND"]

        self.cuantiVar = self.baseOf.select_dtypes(np.number)
        self.cuantiVar = self.cuantiVar.drop(["Población", "AÑO"], axis=1)
        # print(preprocessing.Normalizer().fit_transform(self.cuantiVar))
        # self.cuantiVar = pd.DataFrame(preprocessing.Normalizer().fit_transform(self.cuantiVar),columns=self.cuantiVar.columns.values)
        self.covarianza = self.cuantiVar.cov()
        self.correl = self.cuantiVar.corr()
        val, vec = np.linalg.eig(self.correl)

        features = self.cuantiVar.columns.values

        # Cargando el escalador estandar
        escala = StandardScaler()

        ## Calcula las medias y las desviaciones
        escala.fit(self.cuantiVar)

        # Transforma los datos
        CuantiScale = escala.transform(self.cuantiVar)
        pd.DataFrame(
            CuantiScale, index=self.cuantiVar.index, columns=self.cuantiVar.columns
        )

        # Creamos un objeto PCA y aplicamos
        # Otra opcion pca=PCA(.85), un PCA de dos componenentes
        # 85% minimo valor aceptado (Si es psicometria, esos valores bajan)
        pca = PCA(n_components=2)
        ## Ajuste ese PCA a los datos estandarizados (Calcula los valores y vectores propios)
        pca.fit(CuantiScale)
        # convertimos nuestros datos con las nuevas dimensiones de PCA (calcula las nuevas variables)
        cuanti_pca = pca.transform(CuantiScale)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        PCA_ = []
        for i in range(0, len(pca.components_)):
            x = ["PC", str(i + 1)]
            a = "".join(x)
            PCA_.append(a)

        Pca_Tra = pd.DataFrame(cuanti_pca, index=self.baseOf.index, columns=PCA_)

        ajustPCA = "Ajuste de PCA" + str(CuantiScale.shape)
        expl = pca.explained_variance_ratio_
        expl_ = (
            "Porcentaje de varianza explicada por cada uno de los componentes seleccionados: "
            + str(pca.explained_variance_ratio_)
        )
        resulFin = "% Represantividad de las dimensiones seleccionadas: " + str(
            sum(expl[0:2])
        )

        return Pca_Tra, ajustPCA, expl_, resulFin, features, loadings, cuanti_pca

    def calculoPCATasas(self):
        # Seleccionamos las cuantitativas
        ## Porque PCA solo trabaja con cuantitaivas

        self.baseOf = self.baseOf[self.baseOf["Código DIVIPOLA"] != "ND"]

        self.cuantiVar = self.baseOf.select_dtypes(np.number)

        features = self.cuantiVar.columns.values
        for item, feature in enumerate(features):
            self.cuantiVar[feature] = (
                self.cuantiVar[feature] / self.cuantiVar["Población"]
            )

        self.cuantiVar = self.cuantiVar.drop(["Población", "AÑO"], axis=1)
        self.covarianza = self.cuantiVar.cov()
        self.correl = self.cuantiVar.corr()
        val, vec = np.linalg.eig(self.correl)

        features = self.cuantiVar.columns.values

        # Cargando el escalador estandar
        escala = StandardScaler()

        ## Calcula las medias y las desviaciones
        escala.fit(self.cuantiVar)

        # Transforma los datos
        CuantiScale = escala.transform(self.cuantiVar)
        pd.DataFrame(
            CuantiScale, index=self.cuantiVar.index, columns=self.cuantiVar.columns
        )

        # Creamos un objeto PCA y aplicamos
        # Otra opcion pca=PCA(.85), un PCA de dos componenentes
        # 85% minimo valor aceptado (Si es psicometria, esos valores bajan)
        pca = PCA(n_components=2)
        ## Ajuste ese PCA a los datos estandarizados (Calcula los valores y vectores propios)
        pca.fit(CuantiScale)
        # convertimos nuestros datos con las nuevas dimensiones de PCA (calcula las nuevas variables)
        cuanti_pca = pca.transform(CuantiScale)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        PCA_ = []
        for i in range(0, len(pca.components_)):
            x = ["PC", str(i + 1)]
            a = "".join(x)
            PCA_.append(a)

        Pca_Tra = pd.DataFrame(cuanti_pca, index=self.baseOf.index, columns=PCA_)

        ajustPCA = "Ajuste de PCA" + str(CuantiScale.shape)
        expl = pca.explained_variance_ratio_
        expl_ = (
            "Porcentaje de varianza explicada por cada uno de los componentes seleccionados: "
            + str(pca.explained_variance_ratio_)
        )
        resulFin = "% Represantividad de las dimensiones seleccionadas: " + str(
            sum(expl[0:2])
        )

        return Pca_Tra, ajustPCA, expl_, resulFin, features, loadings, cuanti_pca

    def kmeans(self, nuevosACP, Pca_Tra, n_clusters=3):
        # Se calcula de k de acuerdo a la gráfica del codo
        self.baseOf = self.baseOf[self.baseOf["Código DIVIPOLA"] != "ND"]
        self.baseOf = self.baseOf.drop(["Población", "AÑO"], axis=1)
        kmedias = KMeans(n_clusters=n_clusters).fit(nuevosACP)
        etiquetas = kmedias.labels_

        self.baseOf["Grupo"] = kmedias.labels_
        Pca_Tra["Grupo"] = kmedias.labels_
        matriz = self.baseOf
        tamanho = self.baseOf.groupby("Grupo").size()
        centroides = pd.DataFrame(kmedias.cluster_centers_, columns=["PC1", "PC2"])
        print(centroides)
        return tamanho, centroides, matriz, Pca_Tra, etiquetas
