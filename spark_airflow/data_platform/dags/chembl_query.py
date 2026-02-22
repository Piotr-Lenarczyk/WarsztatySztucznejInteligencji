from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "airflow", "depends_on_past": False}

PARQUET_PATH = "/tmp/chembl_results.parquet"

def export_to_parquet():
    """Export query results to Parquet file"""
    import psycopg2
    import pandas as pd

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="chembl_36",
        user="postgres",
        password="postgres"
    )

    query = """
            SELECT
                cs.canonical_smiles AS smiles,
                a.description AS description,
                a.chembl_id AS assay_chembl_id,
                a.assay_type AS assay_type,
                a.bao_format AS bao_format,
                act.qudt_units AS qudt_units,
                act.activity_id AS activity_id,
                td.pref_name AS target_name,
                act.standard_value AS standard_value,
                act.standard_units AS standard_units,
                act.standard_relation AS standard_relation,
                COALESCE(act.pchembl_value, ROUND(9 - LOG(10, act.standard_value), 2)) AS pchembl_value,
                m.chembl_id AS molecule_chembl_id,
                cp.heavy_atoms AS heavy_atoms,
                (1.37 * COALESCE(act.pchembl_value, ROUND(9 - LOG(10, act.standard_value), 2))) / cp.heavy_atoms AS ligand_efficiency
            FROM compound_structures cs
                RIGHT JOIN molecule_dictionary m ON cs.molregno = m.molregno
                JOIN compound_records r ON m.molregno = r.molregno
                JOIN compound_properties cp ON m.molregno = cp.molregno
                JOIN docs d ON r.doc_id = d.doc_id
                JOIN activities act ON r.record_id = act.record_id
                JOIN assays a ON act.assay_id = a.assay_id
                JOIN target_dictionary td ON a.tid = td.tid
            WHERE act.standard_type = 'IC50'
              AND act.standard_relation = '='
              AND act.standard_units = 'nM'
              AND act.standard_value BETWEEN 0.001 AND 100000
              AND (act.pchembl_value IS NOT NULL OR act.standard_value > 0)
              AND td.organism = 'Homo sapiens'
              AND td.chembl_id = 'CHEMBL203'
              AND m.chembl_id IN
                  (SELECT DISTINCT m1.chembl_id
                   FROM molecule_dictionary m1
                   JOIN molecule_hierarchy mh ON mh.molregno = m1.molregno
                   JOIN molecule_dictionary m2 ON mh.parent_molregno = m2.molregno)
            ORDER BY act.activity_id
            LIMIT 50000
            """

    df = pd.read_sql_query(query, conn)
    print(f"Retrieved {len(df)} records")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved to {PARQUET_PATH}")

    conn.close()


def verify_data():
    """Verify that data was successfully retrieved"""
    import pandas as pd

    df = pd.read_parquet(PARQUET_PATH)

    print("="*80)
    print("DATA VERIFICATION")
    print("="*80)
    print(f"Retrieved {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    if len(df) == 0:
        raise ValueError("No data retrieved from database!")

    print(f"\nData successfully saved to {PARQUET_PATH}")



with DAG(
        dag_id="chembl_query",
        start_date=datetime(2025, 12, 15),
        schedule=None,
        default_args=default_args,
        catchup=False,
        description="Query ChEMBL database and export to Parquet",
        tags=["chembl", "query", "extract"],
) as dag:

    def check_connection():
        """Verify database connection and show statistics"""
        import psycopg2

        conn = psycopg2.connect(
            host="postgres",
            port=5432,
            database="chembl_36",
            user="postgres",
            password="postgres"
        )
        cur = conn.cursor()

        queries = [
            ("Molecules", "SELECT COUNT(*) FROM molecule_dictionary"),
            ("With structures", "SELECT COUNT(*) FROM compound_structures"),
            ("Phase >= 1", "SELECT COUNT(*) FROM molecule_dictionary WHERE max_phase >= 1")
        ]

        print("ChEMBL database statistics:")
        for name, query in queries:
            cur.execute(query)
            count = cur.fetchone()[0]
            print(f"{name}: {count:,}")

        cur.close()
        conn.close()

    # Define tasks
    check_db = PythonOperator(
        task_id="check_connection",
        python_callable=check_connection
    )

    export_data = PythonOperator(
        task_id="export_to_parquet",
        python_callable=export_to_parquet
    )

    verify = PythonOperator(
        task_id="verify_data",
        python_callable=verify_data
    )

    # Define task dependencies
    check_db >> export_data >> verify

