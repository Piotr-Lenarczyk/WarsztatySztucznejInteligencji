from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.bash import BashOperator
import subprocess
import os

default_args = {"owner": "airflow", "depends_on_past": False}

DUMP_DIR = "/opt/airflow/chembl_data"
DUMP_ARCHIVE = f"{DUMP_DIR}/chembl_36_postgresql.tar.gz"
DUMP_FILE = f"{DUMP_DIR}/chembl_36/chembl_36_postgresql/chembl_36_postgresql.dmp"

def check_database_state():
    """Check database state and decide what to do"""

    result = subprocess.run(
        ["psql", "-h", "postgres", "-U", "postgres", "-lqt"],
        env={"PGPASSWORD": "postgres"},
        capture_output=True,
        text=True
    )

    db_exists = "chembl_36" in result.stdout
    print(f"Database chembl_36 exists: {db_exists}")

    if db_exists:
        check_data = subprocess.run(
            [
                "psql", "-h", "postgres", "-U", "postgres",
                "-d", "chembl_36", "-t", "-c",
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
            ],
            env={"PGPASSWORD": "postgres"},
            capture_output=True,
            text=True
        )

        try:
            table_count = int(check_data.stdout.strip())
            print(f"Tables in database: {table_count}")

            if table_count > 10:
                verify_data = subprocess.run(
                    [
                        "psql", "-h", "postgres", "-U", "postgres",
                        "-d", "chembl_36", "-t", "-c",
                        "SELECT COUNT(*) FROM molecule_dictionary;"
                    ],
                    env={"PGPASSWORD": "postgres"},
                    capture_output=True,
                    text=True
                )

                if verify_data.returncode == 0:
                    row_count = int(verify_data.stdout.strip())
                    print(f"Records in molecule_dictionary: {row_count}")

                    if row_count > 0:
                        print("Database is complete - skipping setup")
                        return "all_done"
        except (ValueError, subprocess.CalledProcessError) as e:
            print(f"Error checking data: {e}")

    if not db_exists:
        print("Database does not exist - creating")
        return "create_db"
    else:
        print("Database exists but empty - restoring")
        return "check_dump_exists"

def check_dump_file_exists():
    """Check if dump file was already downloaded and extracted"""

    dump_exists = os.path.exists(DUMP_FILE)
    print(f"Dump exists locally: {dump_exists}")

    if dump_exists:
        size_mb = os.path.getsize(DUMP_FILE) / (1024 * 1024)
        print(f"Dump size: {size_mb:.2f} MB")

        if size_mb > 100:
            print("Dump already downloaded - skipping download")
            return "restore_db"

    archive_exists = os.path.exists(DUMP_ARCHIVE)
    print(f"Archive exists: {archive_exists}")

    if archive_exists:
        print("Archive exists - extracting only")
        return "extract_dump"

    print("No dump found - downloading")
    return "download_dump"



def restore_chembl_with_progress():
    """Restore dump with transaction_timeout filtering"""
    import subprocess
    import tempfile

    original_dump = "/opt/airflow/chembl_data/chembl_36/chembl_36_postgresql/chembl_36_postgresql.dmp"

    print("Starting restore...")

    with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.dmp') as filtered:
        filtered_path = filtered.name

        process = subprocess.Popen(
            ["pg_restore", "-l", original_dump],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        for line in process.stdout:
            if b"SET transaction_timeout" not in line:
                filtered.write(line)

    cmd = [
        "pg_restore",
        "--no-owner",
        "--no-privileges",
        "--verbose",
        "-h", "postgres",
        "-U", "postgres",
        "-d", "chembl_36",
        "--use-list", filtered_path,
        original_dump
    ]

    env = {"PGPASSWORD": "postgres"}

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True,
        bufsize=1
    )

    line_count = 0
    error_count = 0

    for line in iter(process.stdout.readline, ""):
        line_count += 1

        if "error:" in line.lower():
            error_count += 1
            if error_count <= 5:
                print(f"ERROR: {line.strip()}")

        if "restoring data for table" in line.lower():
            print(f"{line.strip()}")

    process.wait()

    print(f"Restore complete - {error_count} errors")

    check_cmd = [
        "psql",
        "-h", "postgres",
        "-U", "postgres",
        "-d", "chembl_36",
        "-c", "SELECT COUNT(*) FROM molecule_dictionary;"
    ]

    result = subprocess.run(
        check_cmd,
        env={"PGPASSWORD": "postgres"},
        capture_output=True,
        text=True
    )

    if result.returncode == 0 and "rows)" in result.stdout:
        print("Import complete")

with DAG(
        dag_id="chembl_setup",
        start_date=datetime(2025, 12, 15),
        schedule=None,
        default_args=default_args,
        catchup=False,
        description="Smart ChEMBL setup - downloads only when needed",
        tags=["chembl", "setup"],
) as dag:

    check_state = BranchPythonOperator(
        task_id="check_database_state",
        python_callable=check_database_state
    )

    all_done = EmptyOperator(
        task_id="all_done"
    )

    create_db = PostgresOperator(
        task_id="create_db",
        postgres_conn_id="postgres_default",
        sql="SELECT 'CREATE DATABASE chembl_36' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'chembl_36')\gexec",
        autocommit=True
    )

    check_dump = BranchPythonOperator(
        task_id="check_dump_exists",
        python_callable=check_dump_file_exists
    )

    download = BashOperator(
        task_id="download_dump",
        bash_command=(
            f"mkdir -p {DUMP_DIR} && "
            f"wget -O {DUMP_ARCHIVE} "
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_postgresql.tar.gz"
        )
    )

    extract = BashOperator(
        task_id="extract_dump",
        bash_command=f"tar -xzf {DUMP_ARCHIVE} -C {DUMP_DIR}",
        trigger_rule="none_failed_min_one_success"
    )

    join_before_restore = EmptyOperator(
        task_id="join_before_restore",
        trigger_rule="none_failed_min_one_success"
    )

    restore_db = PythonOperator(
        task_id="restore_db",
        python_callable=restore_chembl_with_progress,
        execution_timeout=timedelta(hours=2)
    )

    verify = BashOperator(
        task_id="verify_data",
        bash_command=(
            "PGPASSWORD=postgres psql -h postgres -U postgres -d chembl_36 "
            "-c \"SELECT 'Molecules:', COUNT(*) FROM molecule_dictionary; "
            "SELECT 'Assays:', COUNT(*) FROM assays; "
            "SELECT 'Activities:', COUNT(*) FROM activities;\""
        )
    )

    check_state >> all_done
    check_state >> create_db >> check_dump
    check_state >> check_dump
    check_dump >> download >> extract >> join_before_restore
    check_dump >> extract
    extract >> join_before_restore
    check_dump >> join_before_restore
    join_before_restore >> restore_db >> verify
