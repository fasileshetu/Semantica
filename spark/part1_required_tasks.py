from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Part1RequiredTasks") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.option("header", "true").option("inferSchema", "true") \
    .csv("test.csv")

print("Schema:")
df.printSchema()

print("\nSample rows:")
df.show(10)

zip_col = "zipcode"

count_78727 = df.filter(df[zip_col] == 78727).count()
print(f"\nNumber of people in zip code 78727: {count_78727}")

spark.stop()