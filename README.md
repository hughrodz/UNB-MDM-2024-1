# Run Application

```shell
spark-submit \
--master spark://master:7077 \
--executor-memory 32G \
--executor-cores 12 \
topic_modeling_cluster/main.py \
--input=/data/wosdata_lemma_pat.csv \
--sample_rate=1.0 \
--embedding_device=gpu \
--embedding_model=/models/embeddings/all-MiniLM-L6-v2
```
