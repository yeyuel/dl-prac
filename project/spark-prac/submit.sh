spark-submit --master yarn \
--num-executors 4 \
--executor-cores 6 \
--conf spark.yarn.maxAppAttempts=2 \
./hello.py