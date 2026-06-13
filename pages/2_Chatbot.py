import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
from dotenv import load_dotenv
from rdkit.Chem import AllChem

from pipeline.Config import PROJECT_ROOT
from pipelinev2 import predict_pic50_from_smiles, smiles_to_graph, compute_atom_explainability


@st.cache_resource
def load_llm_router():
    from transformers import pipeline

    LOCAL_DIR = PROJECT_ROOT / "local_bart_router"

    if not LOCAL_DIR.exists() or not any(LOCAL_DIR.iterdir()):
        raise FileNotFoundError(
            f"Local router path '{LOCAL_DIR}' is empty. Run 'hf_download.py' first!"
        )

    print(f"Loading Zero-Shot Classifier directly from local disk space: {LOCAL_DIR}")

    classifier = pipeline(
        "zero-shot-classification",
        model=str(LOCAL_DIR),
        tokenizer=str(LOCAL_DIR),
        device=0 if torch.cuda.is_available() else -1
    )
    return classifier

load_dotenv()
llm_router = load_llm_router()


@st.cache_resource
def load_cached_gnn():
    from pipelinev2 import load_model, _load_y_scaler

    # Load GNN model using 34-atom features
    model = load_model(in_channels=34)

    # Extract scaler
    scaler_y = _load_y_scaler()

    return model, scaler_y

# Load cached components globally
gnn_model, target_scaler = load_cached_gnn()

def run_gnn_prediction(smiles: str) -> float:
    # Use the high-level method from pipelinev2 that already handles inference safely
    return predict_pic50_from_smiles(smiles, model=gnn_model, scaler=target_scaler)


# Tokenize and extract valid SMILES strings from user input using RDKit's parsing capabilities
def extract_smiles(text: str) -> Optional[str]:
    # Clean text and split by common delimiters
    clean_text = text.replace(",", " ").replace(";", " ").replace('"', " ").replace("'", " ")
    tokens = clean_text.split()

    for token in tokens:
        # Strip trailing punctuation marks if attached to the string
        token = token.strip(".?!()[]{}")
        if len(token) < 4:
            continue

        # If RDKit can interpret it as a molecule, we found our target!
        mol = Chem.MolFromSmiles(token)
        if mol is not None:
            return token

    return None



def generate_molecule_image(smiles: str) -> Optional[BytesIO]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # Generate canonical 2D coordinates for the atom graph layout
    AllChem.Compute2DCoords(mol)

    # Render the structural image using RDKit's drawing engine
    # options can be added to customize size, atom labeling, or highlights
    img = Draw.MolToImage(mol, size=(300, 300))

    # Save the raw image bytes to an in-memory stream buffer
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


from io import BytesIO
from matplotlib import cm
from rdkit import Chem
from rdkit.Chem import Draw

# Renders a 2D molecule visualization and highlights key atoms driving model prediction
def generate_explainable_molecule_image(smiles: str, atom_weights: Optional[np.ndarray] = None) -> BytesIO:
    mol = Chem.MolFromSmiles(smiles)
    Chem.AllChem.Compute2DCoords(mol)

    highlight_atoms = []
    highlight_colors = {}

    if atom_weights is not None and len(atom_weights) == mol.GetNumAtoms():
        # Map attribution ranges directly to a Matplotlib colormap (e.g., 'Reds' or 'Oranges')
        colormap = cm.get_cmap('Reds')

        for idx, weight in enumerate(atom_weights):
            if weight > 0.3:  # Focus highlights strictly on meaningful attribution nodes
                highlight_atoms.append(idx)
                # Fetch RGB tuple values and assign them to the RDKit index mapping
                rgba = colormap(weight)
                highlight_colors[idx] = (rgba[0], rgba[1], rgba[2])

    # Construct production-grade drawing canvas layout
    drawer = Draw.MolDraw2DCairo(350, 350)

    # Inject drawing specifications safely
    draw_options = drawer.drawOptions()
    draw_options.prepareMolsBeforeDrawing = True

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=None
    )
    drawer.FinishDrawing()

    # Stream binary payload array safely back to Streamlit
    buf = BytesIO(drawer.GetDrawingText())
    buf.seek(0)
    return buf


st.set_page_config(page_title="Chatbot")
st.sidebar.header("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Pass me a chemical SMILES string, and I will parse it to compute its predicted binding potency against Pim-1 kinase."}]

# Render historical chat block
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Catch incoming user input
if user_prompt := st.chat_input("Ask a question or input a SMILES (e.g., c1ccccc1Nc2ncncc2)..."):
    # Render user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Analyzing intent..."):
        # Let the Zero-Shot LLM verify if this is a prediction request
        labels = ["chemical property prediction query", "general conversational greeting or question"]
        route_res = llm_router(user_prompt, candidate_labels=labels)
        top_intent = route_res['labels'][0]

    # Check if a valid chemical string can be pulled via text extraction patterns
    detected_smiles = extract_smiles(user_prompt)

    # Execution path routing logic
    with st.chat_message("assistant"):
        # Initialize response placeholder to guarantee scope safety across all routes
        response = ""

        if top_intent == "chemical property prediction query" or detected_smiles:
            if detected_smiles:
                st.markdown(f"**Chemical Substructure Detected:** `{detected_smiles}`. Processing graph topology...")

                with st.spinner("Assembling molecular graph and computing binding metrics..."):
                    try:
                        # Fetch raw graph target structures
                        g_desc = [0.0, 0.0] # Reconstructed placeholder descriptors logic matching pipelinev2
                        graph_data = smiles_to_graph(detected_smiles, 0.0, g_desc)

                        # Extract explanation attribution scores
                        atom_weights = compute_atom_explainability(gnn_model, graph_data)

                        # Predict the raw potency
                        pIC50 = run_gnn_prediction(detected_smiles)
                        ic50_nm = 10 ** (9 - pIC50)

                        # Draw the heat-mapped 2D structural diagram
                        mol_bytes = generate_explainable_molecule_image(detected_smiles, atom_weights=atom_weights)

                        response = f"""
                        ### 📊 GNN Inference Complete:
                        * **Predicted Potency ($pIC_{{50}}$):** `{pIC50:.3f}`
                        * **Estimated $IC_{{50}}$ Value:** `{ic50_nm:.2f} nM`
                        """
                        st.markdown(response)

                        if mol_bytes:
                            st.image(mol_bytes, caption="XAI Heatmap: Highlighted regions display structural groups driving this prediction.")

                    except Exception as e:
                        response = f"Graph parsing error encountered during target tensor compilation: {str(e)}"
                        st.error(response)
            else:
                response = "I recognized you want a molecular property prediction, but I couldn't parse a valid chemical SMILES string from your text. Please provide an explicit configuration like: `Predict potency for Cc1cnc...`"
                st.markdown(response)
        else:
            # General Chat fallback path
            response = "I recognized this request as a general conversation query. As a dedicated Pim-1 screening agent, you can prompt me with specific molecular configurations to tap into my GNN prediction architecture."
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
