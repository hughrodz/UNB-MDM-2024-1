# Universidade de Brasilia
# Mineracao de Dados Massivos
# Enhancing Bibliometric Analysis with  Parallelized Topic Modeling for Corporate Scalability
#


from sentence_transformers import SentenceTransformer

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

import logging
import time
import argparse

import pandas as pd

import umap
import hdbscan
from gpt4all import GPT4All

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


@udf(returnType=StringType())
def get_label(content):
    model_name = 'Meta-Llama-3-8B-Instruct.Q4_0.gguf'
    model_path = 'models/llm'

    model = GPT4All(model_name=model_name, model_path=model_path, device='cpu')

    system_template = '''Voce é um classificador que recebe uma lista de palavra e identifica o tema relacionado.\n
                         Voce deve ser o mais objetivo possível e responder somente com o nome do tópico usando poucas palavras.
                      '''

    with model.chat_session(system_prompt=system_template):
        resp = model.generate(prompt=content, temp=0)

    return str(resp)


def get_time_elapsed(start_time):
    return str(time.time() - start_time)


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.cluster_id)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def main():

    # Start spark application
    ####################################################################################################################
    logging.info("Start Spark Job - Massive Parallel Document Cluster")

    spark = SparkSession \
        .builder \
        .appName("[UNB/MDM] Massive Parallel Document Cluster") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    sc = spark.sparkContext

    # Load data
    ####################################################################################################################
    load_data_start_time = time.time()
    data = spark.read.options(header=True).csv(args.input).select(col("index").alias("id"),
                                                                  col("TI").alias("title"),
                                                                  col("AB").alias("abstract")).sample(float(args.sample_rate))


    logging.info("Sample rate: " + args.sample_rate)
    logging.info("Total documents: "+str(data.count()))

    # Show data
    ####################################################################################################################
    data.printSchema()
    data.show()
    logging.info("Load data elapsed time: " + get_time_elapsed(load_data_start_time))

    # Embedding process
    ####################################################################################################################
    embedding_start_time = time.time()
    logging.info("Start embedding process using device: "+args.embedding_device)
    logging.info("Broadcast embedding model")
    model = SentenceTransformer(args.embedding_model)
    #model.save('models\\embeddings\\all-MiniLM-L6-v2')
    broadcast_model = spark.sparkContext.broadcast(model)
    embedding_udf = udf(lambda x: broadcast_model.value.encode(x,
                                                               device=args.embedding_device).tolist(),
                                                               ArrayType(FloatType()))

    embedding_df = data.repartition(30).withColumn("embedding", embedding_udf("abstract"))
    logging.info("Cache embedding data")
    embedding_df.cache()


    # Show data embedding
    ####################################################################################################################
    embedding_df.printSchema()
    embedding_df.show()
    logging.info("Embedding elapsed time: " + get_time_elapsed(embedding_start_time))
    logging.info("Total records in Spark dataframe: "+str(embedding_df.count()))

    # UMAP Process
    ####################################################################################################################
    umap_start_time = time.time()
    logging.info("Start UMAP Process")
    logging.info("Convert Spark dataframe to Pandas dataframe using Arrow")
    embedding_df_pandas = embedding_df.select(['id'] + [embedding_df.embedding[x] for x in range(0, 384)]).toPandas()
    print(embedding_df_pandas.columns)
    print(embedding_df_pandas)
    logging.info("Total records in Pandas dataframe: "+str(len(embedding_df_pandas)))

    embedding_df_pandas.set_index('id', inplace=True)
    print(embedding_df_pandas.shape)

    print(embedding_df_pandas.values)

    print(embedding_df_pandas.values)

    umap_embeddings = umap.UMAP(metric='cosine',
                               n_neighbors=15,
                               n_components=5,
                               min_dist=0.0,
                               verbose=True,
                               random_state=42).fit_transform(embedding_df_pandas.values)
    logging.info("UMAP elapsed time: " + get_time_elapsed(umap_start_time))


    # HDBSCAN Process
    ####################################################################################################################
    logging.info("Start HDBSCAN Process")
    umap_pandas_df = pd.DataFrame(umap_embeddings, columns=['feature_1',
                                                            'feature_2',
                                                            'feature_3',
                                                            'feature_4',
                                                            'feature_5'])
    print(umap_pandas_df)
    print(len(umap_pandas_df))

    hdbscan_start_time = time.time()
    cluster = hdbscan.HDBSCAN(min_cluster_size=int(args.min_cluster),
                              metric='euclidean',
                              cluster_selection_method='eom',
                              prediction_data=True).fit(umap_pandas_df.values)

    logging.info("HDBSCAN elapsed time: " + get_time_elapsed(hdbscan_start_time))
    cluster_pandas_df = pd.DataFrame(cluster.labels_, columns=['cluster_id'])

    logging.info("Total cluster: "+str(len(set(cluster.labels_))))

    embedding_df_pandas['id'] = embedding_df_pandas.index
    embedding_df_pandas_2 = embedding_df_pandas.reset_index(drop=True)

    cluster_doc_pandas_df = pd.concat([cluster_pandas_df, embedding_df_pandas_2['id']], axis=1)

    # c-TF-IDF Process
    ####################################################################################################################
    ctfidf_start_time = time.time()
    cluster_spark_df = spark.createDataFrame(cluster_doc_pandas_df)
    cluster_spark_df.show()

    cluster_documents_spark_df = cluster_spark_df.join(data, cluster_spark_df.id == data.id, 'inner').drop(cluster_spark_df.id)
    cluster_documents_spark_df.show()
    cluster_id_text_spark_df = cluster_documents_spark_df.select('cluster_id', 'abstract') \
                                                          .groupBy('cluster_id') \
                                                          .agg(concat_ws(' ', collect_list(col('abstract'))).alias('concat_texts'))

    cluster_id_text_spark_df.show()

    cluster_id_text_pandas_df = cluster_id_text_spark_df.toPandas()

    print(cluster_id_text_pandas_df)

    tf_idf, count = c_tf_idf(cluster_id_text_pandas_df.concat_texts.values, m=len(umap_pandas_df)) # validar
    logging.info("c-TF-IDF elapsed time: " + get_time_elapsed(ctfidf_start_time))

    # Top n Words Process
    ####################################################################################################################
    top_n_start_time = time.time()
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, cluster_id_text_pandas_df, n=10)

    topic_list = []

    for k in top_n_words:
        l = []

        for i in top_n_words[k]:
            l.append(i[0])

        words = ", ".join(str(element) for element in l)

        topic_list.append({
            'cluster_id': str(k),
            'words': words
        })
    logging.info("top n elapsed time: " + get_time_elapsed(top_n_start_time))
    # Topic Label
    ####################################################################################################################
    topic_label_start_time = time.time()
    topics_pandas_df = pd.DataFrame(topic_list)

    topics_spark_df = spark.createDataFrame(topics_pandas_df)

    print('Topic in spark to label')
    topics_spark_df.show()

    logging.info('Compute topic names in spark')
    topics_spark_df.withColumn('topic_label', get_label(col('words'))) \
              .show(truncate=False)


    logging.info("topic label elapsed time: " + get_time_elapsed(topic_label_start_time))
    # Stop Job
    ####################################################################################################################
    logging.info("Stop Spark Job")
    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt='%d/%m/%y %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser(description='Massive Parallel Document Cluster')
    parser.add_argument('--input', action='store', required=True, help='input data')
    parser.add_argument('--sample_rate', action='store', required=False, default=1.0, help='input data')
    parser.add_argument('--output', action='store', required=False, help='output data')
    parser.add_argument('--embedding_device', action='store', required=False, default='cpu', help='optimizer embedding process device')
    parser.add_argument('--embedding_model', action='store', required=True, help='embedding model')
    parser.add_argument('--umap_jobs', action='store', required=False, default=12, help='umap jobs')
    parser.add_argument('--min_cluster', action='store', required=False, default=20, help='hdbscan min cluster size')
    args = parser.parse_args()
    main()
