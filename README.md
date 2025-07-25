# Como segmentar clientes de forma escalável e baseada em dados para estratégias de marketing personalizadas, quando se trabalha com grandes volumes de dados transacionais

# Segmentação Comportamental com Spark (RFMT + PCA + KMeans)
# Dataset: registros transacionais
# Plataforma: Databricks (Spark nativo)

# ============================
# Etapa 1: Leitura e Preparação dos Dados
# ============================

# Leitura do arquivo Parquet direto do DBFS
raw_df = spark.read.parquet("dbfs:/FileStore/data.parquet")

# Conversão da coluna de data para tipo date
from pyspark.sql.functions import col, to_date
raw_df = raw_df.withColumn("dataconsumo", to_date(col("dataconsumo")))

# ============================
# Etapa 2: Engenharia de Atributos RFMT
# ============================

from pyspark.sql.functions import datediff, current_date, mean, stddev, count, min, max, exp, log1p, pow

# Criação de atributos RFMT (Recência, Frequência, Monetário, Ticket)
def calculate_rfmt(df):
    df = df.withColumn("data_diff", datediff(current_date(), col("dataconsumo")))
    df = df.withColumn("recency_score", exp(-0.01 * col("data_diff")))
    df = df.withColumn("ticket", col("quantidade") * col("valorunitario"))
    df = df.withColumn("log_ticket", log1p(col("ticket")))
    df = df.withColumn("qtd2", pow(col("quantidade"), 2))

    agg = df.groupBy("idcliente").agg(
        count("*").alias("frequencia"),
        mean("ticket").alias("ticket_medio"),
        mean("recency_score").alias("recencia"),
        mean("quantidade").alias("qtd_media"),
        stddev("ticket").alias("ticket_std"),
        min("data_diff").alias("dias_ultima_compra")
    )
    return agg

rfmt_df = calculate_rfmt(raw_df)

# ============================
# Etapa 3: Pré-processamento e Vetorização
# ============================

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Vetorização das colunas numéricas
feature_cols = ["frequencia", "ticket_medio", "recencia", "qtd_media", "ticket_std", "dias_ultima_compra"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
pipeline = Pipeline(stages=[assembler, scaler])
model = pipeline.fit(rfmt_df)
scaled_df = model.transform(rfmt_df)

# ============================
# Etapa 4: Redução de Dimensionalidade com PCA
# ============================

from pyspark.ml.feature import PCA

# Aplicação de PCA obrigatório (dados RFMT têm colinearidade)
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(scaled_df)
pca_df = pca_model.transform(scaled_df)

# ============================
# Etapa 5: Clusterização com KMeans
# ============================

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=5, seed=42, featuresCol="pca_features", predictionCol="cluster")
kmeans_model = kmeans.fit(pca_df)
cluster_df = kmeans_model.transform(pca_df)

# ============================
# Etapa 6: Benchmarking e Interpretação
# ============================

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="pca_features", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(cluster_df)
print(f"Silhouette Score (KMeans + PCA): {round(silhouette, 3)}")

# ============================
# Etapa 7: Visualização (opcional)
# ============================

# Para visualização, convertemos para Pandas (caso volume permita)
final_pd = cluster_df.select("idcliente", "cluster").toPandas()

# Exemplo com Seaborn (fora do Spark)
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.countplot(data=final_pd, x="cluster")
# plt.title("Distribuição de Clientes por Cluster")
# plt.show()

# ============================
# Conclusão
# ============================
# ✅ Pipeline completo para clusterização comportamental com Spark
# ✅ PCA essencial para geometria dos vetores RFMT
# ✅ Resultado: Segmentação robusta e pronta para uso em recomendações ou alertas
