import os
import random
import time
from io import BytesIO
from typing import Optional

import streamlit as st
import torch
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from pipelinev2 import predict_pic50_from_smiles


@st.cache_resource
def load_llm_router():
    from transformers import pipeline

    LOCAL_DIR = "./local_bart_router"

    # Verify the files exist before loading
    if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
        raise FileNotFoundError(
            f"Local router path '{LOCAL_DIR}' is empty. Run 'hf_download.py' first!"
        )

    print(f"Loading Zero-Shot Classifier directly from local disk space: {LOCAL_DIR}")

    # Point BOTH the model and tokenizer directly to your offline local folder path
    classifier = pipeline(
        "zero-shot-classification",
        model=LOCAL_DIR,
        tokenizer=LOCAL_DIR,
        device=0 if torch.cuda.is_available() else -1
    )
    return classifier

load_dotenv()
llm_router = load_llm_router()


@st.cache_resource
def load_cached_gnn():
    """
    Loads and caches the compiled model framework and its corresponding
    y-target StandardScaler serialization objects cleanly from disk space.
    """
    from pipelinev2 import load_trained_gnn, _load_y_scaler

    # 1. Load your GNN model using your 34-atom advanced channel dimension
    model = load_trained_gnn(in_channels=34)

    # 2. Extract your production fitted scaler binary matrix
    scaler_y = _load_y_scaler()

    return model, scaler_y

# Load cached components globally
gnn_model, target_scaler = load_cached_gnn()

def run_gnn_prediction(smiles: str) -> float:
    """
    Executes a continuous forward-pass prediction leveraging the pre-loaded
    global GNN network state and target scaling transforms.    """
    # Use the high-level method from pipelinev2 that already handles inference safely
    return predict_pic50_from_smiles(smiles, model=gnn_model, scaler=target_scaler)


def extract_smiles(text: str) -> Optional[str]:
    """
    Robust tokenizer that extracts valid chemical structural features
    from natural language sentences.
    """
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


def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

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
    # Execution path routing logic
    with st.chat_message("assistant"):
        # Initialize response placeholder to guarantee scope safety across all routes
        response = ""

        if top_intent == "chemical property prediction query" or detected_smiles:
            if detected_smiles:
                st.markdown(f"**Chemical Substructure Detected:** `{detected_smiles}`. Processing graph topology...")

                with st.spinner("Assembling molecular graph and computing binding metrics..."):
                    try:
                        pIC50 = run_gnn_prediction(detected_smiles)
                        ic50_nm = 10 ** (9 - pIC50)

                        mol_bytes = generate_molecule_image(detected_smiles)

                        # CHANGED: Assigned directly to 'response' instead of 'response_text'
                        response = f"""
                        ### 📊 GNN Inference Matrix Complete:
                        * **Target Protein Profile:** Serine/threonine-protein kinase Pim-1 (`CHEMBL2147`)
                        * **Predicted Potency ($pIC_{{50}}$):** `{pIC50:.3f}`
                        * **Estimated $IC_{{50}}$ Value:** `{ic50_nm:.2f} nM`
                        """
                        st.markdown(response)

                        # Append the physical 2D image structural layout directly to the chat bubble
                        if mol_bytes:
                            st.image(mol_bytes, caption=f"2D Topological Mapping: {detected_smiles}", use_container_width=False)

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

    # This will now execute perfectly for all conditions without scoping gaps
    st.session_state.messages.append({"role": "assistant", "content": response})
