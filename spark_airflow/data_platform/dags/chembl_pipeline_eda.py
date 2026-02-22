from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "airflow", "depends_on_past": False}

PARQUET_PATH = "/tmp/chembl_results.parquet"
PARQUET_PATH_CLEANED = "/tmp/chembl_results_cleaned.parquet"
PLOTS_DIR = "/opt/airflow/chembl_plots"


def clean_data():
    """Clean the dataset: remove nulls, duplicates, invalid values"""
    import pandas as pd
    from rdkit import Chem

    df = pd.read_parquet(PARQUET_PATH)
    initial_count = len(df)

    print("="*80)
    print("DATA CLEANING")
    print("="*80)
    print(f"Initial records: {initial_count}")

    # 1. Remove rows with missing SMILES (critical for modeling)
    null_smiles = df['smiles'].isnull().sum()
    if null_smiles > 0:
        print(f"Removing {null_smiles} rows with missing SMILES")
        df = df[df['smiles'].notna()]

    # 2. Remove rows with empty or whitespace-only SMILES
    empty_smiles = (df['smiles'].str.strip() == '').sum()
    if empty_smiles > 0:
        print(f"Removing {empty_smiles} rows with empty SMILES")
        df = df[df['smiles'].str.strip() != '']

    # 3. Remove rows with missing pchembl_value (target variable)
    null_pchembl = df['pchembl_value'].isnull().sum()
    if null_pchembl > 0:
        print(f"Removing {null_pchembl} rows with missing pchembl_value")
        df = df[df['pchembl_value'].notna()]

    # 4. Remove rows with missing or invalid standard_value
    null_std_val = df['standard_value'].isnull().sum()
    if null_std_val > 0:
        print(f"Removing {null_std_val} rows with missing standard_value")
        df = df[df['standard_value'].notna()]

    invalid_std_val = (df['standard_value'] <= 0).sum()
    if invalid_std_val > 0:
        print(f"Removing {invalid_std_val} rows with invalid standard_value (<=0)")
        df = df[df['standard_value'] > 0]

    # 5. Remove duplicate activity records
    duplicates = df.duplicated(subset=['activity_id']).sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate activities")
        df = df.drop_duplicates(subset=['activity_id'])

    # 6. Remove duplicate compound-assay combinations (keep first)
    compound_assay_dups = df.duplicated(subset=['smiles', 'assay_chembl_id']).sum()
    if compound_assay_dups > 0:
        print(f"Removing {compound_assay_dups} duplicate compound-assay combinations")
        df = df.drop_duplicates(subset=['smiles', 'assay_chembl_id'], keep='first')

    # 7. Verify SMILES validity
    print("Validating SMILES structures...")
    invalid_smiles = 0
    valid_mask = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        valid_mask.append(mol is not None)
        if mol is None:
            invalid_smiles += 1

    if invalid_smiles > 0:
        print(f"Removing {invalid_smiles} rows with invalid SMILES structures")
        df = df[valid_mask]

    # 8. Reset index
    df = df.reset_index(drop=True)

    final_count = len(df)
    removed = initial_count - final_count
    removed_pct = (removed / initial_count * 100) if initial_count > 0 else 0

    print(f"\nCleaning summary:")
    print(f"  Initial records: {initial_count}")
    print(f"  Final records: {final_count}")
    print(f"  Removed: {removed} ({removed_pct:.2f}%)")
    print(f"  Data quality: {(final_count/initial_count*100):.2f}%")

    # Save cleaned data
    df.to_parquet(PARQUET_PATH_CLEANED, index=False)
    print(f"\nCleaned data saved to {PARQUET_PATH_CLEANED}")

    if final_count == 0:
        raise ValueError("All records were removed during cleaning!")


def correlation_analysis():
    """Analyze correlations and remove highly correlated features (>0.8)"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH_CLEANED)

    print("="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    print(f"Original DataFrame shape: {df.shape}")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr().abs()

    # Create a mask for upper triangle
    upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    upper_corr = correlation_matrix.where(upper_triangle)

    # Find features with correlation > 0.8
    to_drop = [column for column in upper_corr.columns if any(upper_corr[column] > 0.8)]

    print(f"\nFeatures to drop due to high correlation (> 0.8): {len(to_drop)}")
    if to_drop:
        print(f"Dropping: {to_drop}")

    # Visualize correlation heatmap before dropping
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap (Before Dropping)')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_correlation_heatmap.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nCorrelation heatmap saved to {PLOTS_DIR}/01_correlation_heatmap.png")

    # Drop highly correlated features
    df_cleaned = df.drop(columns=to_drop)
    print(f"DataFrame shape after dropping correlated features: {df_cleaned.shape}")
    print(f"Removed {len(to_drop)} features")

    # Save cleaned data
    df_cleaned.to_parquet(PARQUET_PATH_CLEANED, index=False)
    print(f"\nUpdated cleaned data saved to {PARQUET_PATH_CLEANED}")


def null_values_analysis():
    """Analyze null values in the dataset"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH_CLEANED)

    print("="*80)
    print("NULL VALUES ANALYSIS")
    print("="*80)

    null_columns = {}
    for column in df.columns:
        null_count = df[column].isnull().sum()
        null_columns[column] = null_count

    null_counts = pd.Series(null_columns)
    total_nulls = null_counts.sum()

    if total_nulls > 0:
        print(f"\nTotal null values: {total_nulls}")
        print("\nNull values by column:")
        for col, count in null_counts[null_counts > 0].items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")

        # Visualize null values
        plt.figure(figsize=(12, 6))
        null_counts.sort_values(ascending=False).plot(kind='bar')
        plt.title('Null Value Count by Column')
        plt.xlabel('Column')
        plt.ylabel('Null Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/05_null_values.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"\nNull values plot saved to {PLOTS_DIR}/05_null_values.png")
    else:
        print("\nNo null values found in the DataFrame")


def distribution_analysis():
    """Analyze distribution of IC50 and pIC50 with visualizations"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH_CLEANED)

    print("="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)

    # Rename columns for easier handling
    df_dist = df[["smiles", "molecule_chembl_id", "standard_value",
                  "standard_units", "standard_relation", "pchembl_value"]].copy()
    df_dist = df_dist.rename(columns={'standard_value': 'IC50', 'pchembl_value': 'pIC50'})

    print(f"\nDataset shape: {df_dist.shape}")
    print("\nStandard units distribution:")
    print(df_dist["standard_units"].value_counts())
    print("\nStandard relation distribution:")
    print(df_dist["standard_relation"].value_counts())

    # Create distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(df_dist["IC50"], bins=50, ax=axes[0], kde=True)
    axes[0].set_title("Distribution of Standard Values")
    axes[0].set_xlabel("Standard Values IC50 (nM)")
    axes[0].set_ylabel("Count")

    sns.histplot(df_dist["pIC50"], bins=50, ax=axes[1], kde=True)
    axes[1].set_title("Distribution of pChEMBL Values")
    axes[1].set_xlabel("pChEMBL Values pIC50 (-log10)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_distribution_analysis.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nDistribution plots saved to {PLOTS_DIR}/02_distribution_analysis.png")


def outlier_analysis():
    """Analyze and visualize outliers in pIC50"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH_CLEANED)

    print("="*80)
    print("OUTLIER ANALYSIS")
    print("="*80)

    df_eda = df[["smiles", "molecule_chembl_id", "standard_value", "standard_units",
                 "standard_relation", "pchembl_value", "description"]].copy()
    df_eda.dropna(subset=["smiles", "standard_value", "pchembl_value"], inplace=True)
    df_eda = df_eda.rename(columns={'standard_value': 'IC50', 'pchembl_value': 'pIC50'})

    print(f"Dataset shape: {df_eda.shape}")

    # Create visualization with histograms and boxplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(df_eda["IC50"], bins=50, ax=axes[0, 0], kde=True, log_scale=True)
    axes[0, 0].set_title("Histogram of IC50")
    axes[0, 0].set_xlabel("IC50 (nM) - Logarithmic Scale")
    axes[0, 0].set_ylabel("Count")

    sns.histplot(df_eda["pIC50"], bins=50, ax=axes[0, 1], kde=False)
    axes[0, 1].set_title("Histogram of pIC50")
    axes[0, 1].set_xlabel("pIC50")
    axes[0, 1].set_ylabel("Count")

    sns.boxplot(x=df_eda["IC50"], ax=axes[1, 0])
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title("Boxplot of IC50 - outliers")

    sns.boxplot(x=df_eda["pIC50"], ax=axes[1, 1])
    axes[1, 1].set_title("Boxplot of pIC50 - outliers")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_outlier_analysis.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nOutlier plots saved to {PLOTS_DIR}/03_outlier_analysis.png")

    # Calculate outlier threshold
    q25 = df_eda["pIC50"].quantile(0.25)
    q75 = df_eda["pIC50"].quantile(0.75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr

    print(f"\nOutlier threshold (IQR method): {upper_bound:.2f}")

    outliers_count = (df_eda["pIC50"] > upper_bound).sum()
    print(f"Number of outliers: {outliers_count} ({outliers_count/len(df_eda)*100:.2f}%)")

    df_no_outliers = df_eda[df_eda["pIC50"] <= upper_bound]
    print(f"\nDataset shape after removing outliers: {df_no_outliers.shape}")
    print(f"Removed {len(df_eda) - len(df_no_outliers)} outliers")

    print(f"\nStatistics without outliers:")
    print(f"  Max pIC50: {df_no_outliers['pIC50'].max():.2f}")
    print(f"  Min pIC50: {df_no_outliers['pIC50'].min():.2f}")
    print(f"  Mean pIC50: {df_no_outliers['pIC50'].mean():.2f}")
    print(f"  Median pIC50: {df_no_outliers['pIC50'].median():.2f}")


def scaffold_analysis():
    """Analyze Bemis-Murcko scaffolds"""
    import pandas as pd
    import os
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    df = pd.read_parquet(PARQUET_PATH_CLEANED)

    print("="*80)
    print("BEMIS-MURCKO SCAFFOLD ANALYSIS")
    print("="*80)

    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        return None

    scaffolds = df[["smiles", "pchembl_value"]].copy()
    scaffolds = scaffolds.rename(columns={'pchembl_value': 'pIC50'})
    print(f"Processing {len(scaffolds)} molecules...")

    scaffolds["scaffold"] = scaffolds["smiles"].apply(get_scaffold)

    # Check how many times the scaffold appears and its average pIC50
    scaffold_stats = scaffolds.groupby("scaffold").agg(
        count=("pIC50", "count"),
        avg_pIC50=("pIC50", "mean")
    ).reset_index()

    print(f"\nUnique scaffolds: {scaffold_stats.shape[0]}")
    print(f"Average pIC50 for a single scaffold: {scaffold_stats['avg_pIC50'].mean():.2f}")

    # Top 10 scaffolds by count
    top_scaffolds = scaffold_stats.sort_values(by="count", ascending=False).head(10)
    print("\nTop 10 most common scaffolds:")
    for i, row in enumerate(top_scaffolds.itertuples(), 1):
        print(f"  {i}. Count: {row.count}, Avg pIC50: {row.avg_pIC50:.2f}")
        print(f"     SMILES: {row.scaffold[:60]}...")

    # Note: Molecular structure images cannot be easily generated in Airflow without display
    # For visualization, use the notebook or a separate image generation task
    print(f"\nScaffold analysis complete. For visual scaffold structures, use notebook.")


def lipinski_analysis():
    """Analyze Lipinski's Rule of Five compliance with visualizations"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH_CLEANED)

    print("="*80)
    print("LIPINSKI'S RULE OF FIVE ANALYSIS")
    print("="*80)

    def calculate_lipinski(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            return {"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba}
        return {"MW": None, "LogP": None, "HBD": None, "HBA": None}

    df_eda = df[["smiles", "molecule_chembl_id", "standard_value", "standard_units",
                 "standard_relation", "pchembl_value", "description"]].copy()
    df_eda.dropna(subset=["smiles", "standard_value", "pchembl_value"], inplace=True)
    df_eda = df_eda.rename(columns={'standard_value': 'IC50', 'pchembl_value': 'pIC50'})

    print(f"Calculating Lipinski descriptors for {len(df_eda)} molecules...")

    lipinski_results = df_eda["smiles"].apply(calculate_lipinski)
    lipinski_df = pd.DataFrame(lipinski_results.tolist())
    df_lipinski = pd.concat([df_eda, lipinski_df], axis=1)

    # Calculate violations
    violations = {
        "MW": (df_lipinski["MW"] > 500).sum(),
        "LogP": (df_lipinski["LogP"] > 5).sum(),
        "HBD": (df_lipinski["HBD"] > 5).sum(),
        "HBA": (df_lipinski["HBA"] > 10).sum()
    }

    print("\nLipinski's Rule violations:")
    for rule, count in violations.items():
        print(f"  {rule}: {count} violations ({count/len(df_lipinski)*100:.2f}%)")

    # Calculate total violations per molecule
    viol_mask = pd.DataFrame({
        "MW": df_lipinski["MW"] > 500,
        "LogP": df_lipinski["LogP"] > 5,
        "HBD": df_lipinski["HBD"] > 5,
        "HBA": df_lipinski["HBA"] > 10
    }).fillna(False)

    df_lipinski["lipinski_violations"] = viol_mask.sum(axis=1).astype(int)
    violation_counts = df_lipinski["lipinski_violations"].value_counts().sort_index()
    counts_list = [int(violation_counts.get(i, 0)) for i in range(5)]

    print("\nViolation distribution:")
    for i, count in enumerate(counts_list):
        print(f"  {i} violations: {count} ({count/len(df_lipinski)*100:.2f}%)")

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # MW vs LogP
    sns.scatterplot(data=df_lipinski, x="MW", y="LogP", hue="pIC50",
                    palette="viridis", ax=axes[0, 0], legend=False)
    axes[0, 0].axvline(500, color='red', linestyle='--', label='MW = 500')
    axes[0, 0].axhline(5, color='blue', linestyle='--', label='LogP = 5')
    axes[0, 0].set_title("Molecular Weight vs LogP")
    axes[0, 0].legend()

    # Violation counts
    sns.barplot(x=list(range(5)), y=counts_list, hue=list(range(5)),
                palette="viridis", ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("Count of Lipinski's Rule Violations")
    axes[0, 1].set_xlabel("Number of Violations")
    axes[0, 1].set_ylabel("Count of Compounds")

    # MW vs pIC50
    sns.scatterplot(data=df_lipinski, x="MW", y="pIC50", hue="pIC50",
                    palette="viridis", ax=axes[1, 0], legend=False)
    axes[1, 0].axvline(500, color='red', linestyle='--', label='MW = 500')
    axes[1, 0].set_title("Molecular Weight vs pIC50")
    axes[1, 0].legend()

    # LogP vs pIC50
    sns.scatterplot(data=df_lipinski, x="LogP", y="pIC50", hue="pIC50",
                    palette="viridis", ax=axes[1, 1], legend=False)
    axes[1, 1].axvline(5, color='red', linestyle='--', label='LogP = 5')
    axes[1, 1].set_title("LogP vs pIC50")
    axes[1, 1].legend()

    # HBD vs HBA
    sns.scatterplot(data=df_lipinski, x="HBD", y="HBA", hue="pIC50",
                    palette="viridis", ax=axes[2, 0], legend=False)
    axes[2, 0].axvline(5, color='red', linestyle='--', label='HBD = 5')
    axes[2, 0].axhline(10, color='blue', linestyle='--', label='HBA = 10')
    axes[2, 0].set_title("Hydrogen Bond Donors vs Acceptors")
    axes[2, 0].legend()

    # Hide the empty subplot
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_lipinski_analysis.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nLipinski analysis plots saved to {PLOTS_DIR}/04_lipinski_analysis.png")


with DAG(
        dag_id="chembl_pipeline_eda",
        start_date=datetime(2025, 12, 15),
        schedule=None,
        default_args=default_args,
        catchup=False,
        description="Complete pipeline EDA: cleaning, correlation, distribution, outliers, scaffold, Lipinski",
        tags=["chembl", "eda", "pipeline", "lipinski", "scaffold"],
) as dag:

    clean_task = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data
    )

    correlation_task = PythonOperator(
        task_id="correlation_analysis",
        python_callable=correlation_analysis
    )

    null_values_task = PythonOperator(
        task_id="null_values_analysis",
        python_callable=null_values_analysis
    )

    distribution_task = PythonOperator(
        task_id="distribution_analysis",
        python_callable=distribution_analysis
    )

    outlier_task = PythonOperator(
        task_id="outlier_analysis",
        python_callable=outlier_analysis
    )

    scaffold_task = PythonOperator(
        task_id="scaffold_analysis",
        python_callable=scaffold_analysis
    )

    lipinski_task = PythonOperator(
        task_id="lipinski_analysis",
        python_callable=lipinski_analysis
    )

    # Define dependencies following notebook structure:
    # 1. Clean data first
    # 2. Analyze correlations (may remove columns)
    # 3. Check null values
    # 4. Distribution analysis
    # 5. Parallel: outlier, scaffold, lipinski

    clean_task >> correlation_task >> null_values_task >> distribution_task
    distribution_task >> [outlier_task, scaffold_task, lipinski_task]

