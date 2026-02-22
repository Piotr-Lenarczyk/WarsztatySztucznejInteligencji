import argparse
import sys
from pyspark.sql import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument("--jdbc-url", required=True)
parser.add_argument("--user", required=True)
parser.add_argument("--password", required=True)
parser.add_argument("--output-path", default="/tmp/chembl_results.parquet")
parser.add_argument("--limit", type=int, default=1000)
args = parser.parse_args()

try:
    spark = SparkSession.builder \
        .appName("chembl_query") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3") \
        .getOrCreate()

    db_table = f"""(SELECT
    cs.canonical_smiles AS smiles,
    r.compound_key AS compound_key,
    a.description AS description,
    a.chembl_id AS chembl_id,
    td.pref_name AS target_name,
    td.target_type AS target_type,
    act.standard_type AS standard_type,
    act.standard_value AS standard_value,
    act.standard_units AS standard_units,
    act.standard_relation AS standard_relation,
    COALESCE(act.pchembl_value, ROUND(9 - LOG(10, act.standard_value), 2)) AS pchembl_value,
    act.activity_id AS activity_id,
    act.assay_id AS assay_id,
    td.organism AS target_organism
    FROM compound_structures cs
    RIGHT JOIN molecule_dictionary m ON cs.molregno = m.molregno
    JOIN compound_records r ON m.molregno = r.molregno
    JOIN docs d ON r.doc_id = d.doc_id
    JOIN activities act ON r.record_id = act.record_id
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    WHERE act.standard_type = 'IC50'
    AND act.standard_relation = '='
    AND act.standard_units = 'nM'
    AND (act.pchembl_value IS NOT NULL OR act.standard_value > 0)
    AND td.organism = 'Homo sapiens'
    LIMIT {args.limit}) as q"""

    print(f"Connecting to: {args.jdbc_url}")

    df = spark.read.format("jdbc") \
        .option("url", args.jdbc_url) \
        .option("dbtable", db_table) \
        .option("user", args.user) \
        .option("password", args.password) \
        .option("driver", "org.postgresql.Driver") \
        .option("fetchsize", "1000") \
        .load()

    print(f"Retrieved {df.count()} records")

    df.show(5, truncate=False)

    print(f"Saving to: {args.output_path}")
    df.write.mode("overwrite").parquet(args.output_path)

    print("Job completed successfully")
    spark.stop()

except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
