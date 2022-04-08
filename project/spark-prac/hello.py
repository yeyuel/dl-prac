import pyspark
import findspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.\
        master('yarn').\
        config('spark.executor.memory', '4g').\
        config('spark.executor.cores', '4').\
        config('spark.driver.memory','4g').\
        config('spark.dynamicAllocation.minExecutors', '10').\
        config('spark.dynamicAllocation.maxExecutors', '50').\
        config('spark.executor.instances', 6).\
        config('spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version', '2').\
        config('spark.sql.execution.arrow.enabled', 'false').\
        config('spark.driver.maxResultSize', '5g').\
        config('spark.sql.shuffle.partitions', '2000').\
        config('spark.default.parallelism', '50').\
        config('hive.exec.orc.split.strategy', 'ETL').\
        appName('movie_lens_feature').\
        enableHiveSupport().getOrCreate()
sc = spark.sparkContext

print("spark version: ", pyspark.__version__)
rdd = sc.parallelize(["hello", "spark"])
print(rdd.reduce(lambda x, y: x + ' ' + y))
