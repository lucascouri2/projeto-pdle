#########################################################################################
# Alunos:																				#
# 	Laianna Lana Virginio da Silva - llvs2@cin.ufpe.br      							#
# 	Lucas Natan Correia Couri - lncc2@cin.ufpe.br   									#
# 	Priscilla Amarante de Lima - pal4@cin.ufpe.br   									#
#########################################################################################

#########################################################################################
# Bibliotecas																			#
#########################################################################################
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml import Pipeline
from pyspark.shell import spark

#########################################################################################
SEED = 42
#########################################################################################


#########################################################################################
# Carrega o arquivo
pt7_parquet = spark.read.format("parquet").load(f"/bigdata/pt7-hash.parquet")

# Indexa as labels
label_indexador = StringIndexer(inputCol = "label", outputCol = "indexedLabel").fit(pt7_parquet)

# Divide o dataset em treino e teste
a, b = pt7_parquet.randomSplit(weights = [0.01, 0.3], seed = SEED)
treino, teste = a.randomSplit(weights = [0.7, 0.3], seed = SEED)

# modelo Random Forest
rf = RandomForestClassifier(labelCol = "indexedLabel", featuresCol = "features", numTrees = 5)

# Converte as labels int de volta ao original
label_conversor = IndexToString(inputCol = "prediction", outputCol = "predictedLabel", labels = label_indexador.labels)

# Faz o Pipeline
pipeline = Pipeline(stages = [label_indexador, rf, label_conversor])

# Treina o modelo
modelo = pipeline.fit(treino)

# Faz a predição
predicao = modelo.transform(teste)


#########################################################################################
# Métricas de Avaliação																	#
#########################################################################################
# Acurácia
avaliador_acuracia = MulticlassClassificationEvaluator(labelCol = "indexedLabel", predictionCol = "prediction", metricName = "accuracy")
acuracia = avaliador_acuracia.evaluate(predicao)

# F1
evaluator_f1 = MulticlassClassificationEvaluator(labelCol = "indexedLabel", predictionCol = "prediction", metricName = "f1")
f1 = evaluator_f1.evaluate(predicao)

# Precisão
evaluator_precisao_label = MulticlassClassificationEvaluator(labelCol = "indexedLabel", predictionCol = "prediction", metricName = "precisionByLabel")
precisao_label = evaluator_precisao_label.evaluate(predicao)

# Precisão
evaluator_precisao_ponderada = MulticlassClassificationEvaluator(labelCol = "indexedLabel", predictionCol = "prediction", metricName = "weightedPrecision")
precisao_ponderada = evaluator_precisao_ponderada.evaluate(predicao)

# Recall
evaluator_recall_label = MulticlassClassificationEvaluator(labelCol = "indexedLabel", predictionCol = "prediction", metricName = "recallByLabel")
recall_label = evaluator_recall_label.evaluate(predicao)

# Recall
evaluator_recall_ponderado = MulticlassClassificationEvaluator(labelCol = "indexedLabel", predictionCol = "prediction", metricName = "weightedRecall")
recall_ponderado = evaluator_recall_ponderado.evaluate(predicao)


#########################################################################################
# Salva o modelo_rf e as métricas de avaliação											#
#########################################################################################
#salva o modelo rf
#modelo.save(f"/projeto/modelo_rf")
modelo.save(f"hdfs://master:8020/bigdata/modelo_rf")

#salva as métricas
arquivo = open("projeto/metricas.txt", "w")
arquivo.write(f"acucacia,f1,precisao_label,precisao_ponderada,recall_label,recall_ponderado\n{acuracia},{f1},{precisao_label},{precisao_ponderada},{recall_label},{recall_ponderado}")
arquivo.close()