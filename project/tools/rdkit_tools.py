# tools/rdkit_tools.py

from crewai_tools import BaseTool
from pydantic import Field
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import SimilarityMaps
import streamlit as st
import base64
from io import BytesIO
from rdkit.Chem.Draw import rdMolDraw2D


class CalculateLogPTool(BaseTool):
    name: str = Field(default="Calculate LogP")
    description: str = Field(default="Calculates the LogP of the compound.")

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid SMILES string"
            logp = Descriptors.MolLogP(mol)
            return f"LogP: {logp}"
        except Exception as e:
            print(f"Error calculating LogP: {str(e)}")
            return f"Error calculating LogP: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class CalculateMolWtTool(BaseTool):
    name: str = Field(default="Calculate Molecular Weight")
    description: str = Field(default="Calculates the molecular weight of the compound.")

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid SMILES string"
            mol_wt = Descriptors.ExactMolWt(mol)
            return f"Molecular Weight: {mol_wt}"
        except Exception as e:
            print(f"Error calculating Molecular Weight: {str(e)}")
            return f"Error calculating Molecular Weight: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class LipinskiRuleOfFiveTool(BaseTool):
    name: str = Field(default="Lipinski Rule of Five")
    description: str = Field(default="Evaluates Lipinski's Rule of Five for druglikeness.")

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid SMILES string"

            MW = Descriptors.MolWt(mol)
            HBA = rdMolDescriptors.CalcNumHBA(mol)
            HBD = rdMolDescriptors.CalcNumHBD(mol)
            LogP = Descriptors.MolLogP(mol)

            conditions = [MW <= 500, HBA <= 10, HBD <= 5, LogP <= 5]
            violations = 4 - conditions.count(True)

            result = f"Lipinski Rule of Five:\n"
            result += f" - Molecular Weight: {MW} Da\n"
            result += f" - Hydrogen Bond Acceptors: {HBA}\n"
            result += f" - Hydrogen Bond Donors: {HBD}\n"
            result += f" - LogP: {LogP}\n"
            result += f" - Violations: {violations}\n"
            result += "Compound is likely to be orally active." if violations <= 1 else "Compound is unlikely to be orally active."

            return result
        except Exception as e:
            print(f"Error evaluating Lipinski's Rule: {str(e)}")
            return f"Error evaluating Lipinski's Rule: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class SimilarityMapTool(BaseTool):
    name: str = Field(default="Generate Similarity Map")
    description: str = Field(default="Generates a similarity map between two molecules.")

    def _run(self, smiles1: str, smiles2: str) -> str:
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                return "Invalid SMILES string(s)" 
            
            # Create a drawing object
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)

            # Generate the similarity map
            SimilarityMaps.GetSimilarityMapForFingerprint(
                mol1, mol2, SimilarityMaps.GetMorganFingerprint, draw2d=drawer
            )
            
            # Finish drawing and get the image data
            drawer.FinishDrawing()
            image = drawer.GetDrawingText()

            # Convert image data to base64 string
            img_str = base64.b64encode(image).decode()
            
            # Display image in Streamlit
            html_img = f'<img src="data:image/png;base64,{img_str}" alt="Similarity Map"/>'
            st.markdown(html_img, unsafe_allow_html=True)

            return "Similarity map displayed."
        except Exception as e:
            print(f"Error generating similarity map: {str(e)}")
            return f"Error generating similarity map: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

        
# class TanimotoSimilarity(BaseTool):
    