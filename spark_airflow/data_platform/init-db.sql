-- Automatyczne tworzenie bazy danych dla Airflow
-- Ten skrypt uruchamia się przy pierwszym starcie PostgreSQL

-- Tworzenie bazy danych dla Airflow
SELECT 'CREATE DATABASE airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

-- Alternatywnie możesz użyć prostszej składni (ale wymaga uprawnień superuser):
-- CREATE DATABASE IF NOT EXISTS airflow;

-- Możesz także utworzyć dodatkowe bazy dla swoich danych:
-- CREATE DATABASE IF NOT EXISTS my_data;

-- Informacja o utworzonych bazach
\echo 'Bazy danych PostgreSQL:'
\l

