from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Part1RequiredTasks") \
    .master("local[*]") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider") \
    .config("spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .getOrCreate()

df = spark.read.option("header", "true").option("inferSchema", "true") \
    .csv("s3a://vagelis-testbucket1/test.csv")