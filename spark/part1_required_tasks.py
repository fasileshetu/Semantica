"""
Part 1 required tasks:
1c. Create a DataFrame from s3://vagelis-testbucket1/test.csv
    Write a query to find how many people live in zip code 78727

Uses your AWS credentials (already configured via `aws configure` on this
machine) rather than anonymous access, since the bucket requires auth.
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Part1RequiredTasks") \
    .master("local[*]") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.profile.ProfileCredentialsProvider") \
    .config("spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .getOrCreate()

# --- Read the test CSV from S3 ---
df = spark.read.option("header", "true").option("inferSchema", "true") \
    .csv("s3a://vagelis-testbucket1/test.csv")

print("Schema:")
df.printSchema()

print("\nSample rows:")
df.show(10)

# --- Required query: how many people live in zip code 78727 ---
# NOTE: adjust the column name below once you see the actual schema above —
# it may be "zip", "zipcode", "zip_code", etc.
zip_col = "zip"  # <-- update this after checking df.printSchema() output

count_78727 = df.filter(df[zip_col] == 78727).count()
print(f"\nNumber of people in zip code 78727: {count_78727}")

spark.stop()