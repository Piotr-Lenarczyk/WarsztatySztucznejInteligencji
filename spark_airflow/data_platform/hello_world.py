from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
data = ["Hello", "World"]
df = spark.createDataFrame([(x,) for x in data], ["word"])
df.show()
spark.stop()