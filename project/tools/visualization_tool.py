# tools/visualization_tool.py

from crewai_tools import BaseTool
from pydantic import Field
from rdkit import Chem
from rdkit.Chem import Draw
import streamlit as st
import base64
from io import BytesIO

class VisualizationTool(BaseTool):
    name: str = Field(default="Molecule Visualization Tool")
    description: str = Field(default="Generates an image of the molecule from a SMILES string")

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid molecule for visualization."

            # Generate the image
            img = Draw.MolToImage(mol, size=(300, 300))

            # Convert image to base64 string
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Return an HTML image tag
            html_img = f'<img src="data:image/png;base64,{img_str}" alt="Molecule Image"/>'

            # Since Streamlit cannot render HTML in markdown, we use components
            st.markdown(html_img, unsafe_allow_html=True)

            return "Molecule visualization displayed."
        except Exception as e:
            print(f"Error in molecule visualization: {str(e)}")
            return f"Error in molecule visualization: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
