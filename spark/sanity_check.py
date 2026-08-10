"""
Minimal Spark sanity check: parse sample_articles.xml, load into a DataFrame,
run a basic transformation. This confirms the Spark install works end-to-end
before scaling up to the real Wikipedia dump.
"""

import xml.etree.ElementTree as ET
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, col

# --- Parse the XML into plain (title, extract) tuples ---
tree = ET.parse("sample_articles.xml")
root = tree.getroot()

records = []
for page in root.iter("page"):
    title = page.get("title")
    extract_el = page.find("extract")
    extract = extract_el.text if extract_el is not None else ""
    if extract:
        records.append((title, extract.strip()))

print(f"Parsed {len(records)} articles from sample_articles.xml")

# --- Spark ---
spark = SparkSession.builder \
    .appName("SemanticaSanityCheck") \
    .master("local[*]") \
    .getOrCreate()

df = spark.createDataFrame(records, ["title", "extract"])

print("\nSchema:")
df.printSchema()

print("\nSample rows:")
df.show(5, truncate=80)

# Basic transformation: article length, sorted longest first
df_with_length = df.withColumn("extract_length", length(col("extract")))
print("\nArticles sorted by extract length (descending):")
df_with_length.orderBy(col("extract_length").desc()).select("title", "extract_length").show(20, truncate=False)

spark.stop()