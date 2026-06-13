import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

from pipeline.Config import setup_dirs, Config


def run_full_eda(df: pd.DataFrame):
    """Run standard exploratory data analysis and generate plots."""
    setup_dirs()
    print("--- Running EDA ---")

    # Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr().abs(), cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Analysis Matrix')
        plt.savefig(str(Config.plots_dir / "eda_correlation.png"))
        plt.close()

    # Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["standard_value"], bins=50, ax=axes[0], kde=True, log_scale=True)
    axes[0].set_title("Raw IC50 (nM) Log-Scale Distribution")
    sns.histplot(df["pchembl_value"], bins=50, ax=axes[1], kde=True)
    axes[1].set_title("pIC50 Distribution")
    plt.savefig(str(Config.plots_dir / "eda_distributions.png"))
    plt.close()

    # Lipinski's Rule of 5 properties
    def calc_lipinski(smi):
        mol = Chem.MolFromSmiles(smi)
        if not mol: return [None]*4
        return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]

    lip_cols = ["MW", "LogP", "HBD", "HBA"]
    df[lip_cols] = pd.DataFrame(df["smiles"].apply(calc_lipinski).tolist(), index=df.index)
    df["lipinski_violations"] = (df["MW"] > 500).astype(int) + (df["LogP"] > 5).astype(int) + \
                                (df["HBD"] > 5).astype(int) + (df["HBA"] > 10).astype(int)

    plt.figure(figsize=(8, 5))
    sns.countplot(x="lipinski_violations", data=df, palette="viridis", hue="lipinski_violations", legend=False)
    plt.title("Lipinski Rules Structural Violations")
    plt.savefig(str(Config.plots_dir / "eda_lipinski_violations.png"))
    plt.close()

    # Scaffold Analysis
    df["scaffold"] = df["smiles"].apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
    scaff_counts = df["scaffold"].value_counts().head(10)

    mols = [Chem.MolFromSmiles(s) for s in scaff_counts.index if s]
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, legends=[f"Count: {c}" for c in scaff_counts.values])
        img.save(str(Config.plots_dir / "eda_top_scaffolds.png"))