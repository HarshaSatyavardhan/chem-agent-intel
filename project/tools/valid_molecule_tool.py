# tools/valid_molecule_tool.py

from crewai_tools import BaseTool
from pydantic import Field
from rdkit import Chem

class ValidMoleculeTool(BaseTool):
    name: str = Field(default="Valid Molecule Checker")
    description: str = Field(default="Checks if a SMILES string is a valid molecule")

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid"
            return "Valid"
        except Exception as e:
            print(f"Error validating molecule: {str(e)}")
            return f"Error validating molecule: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
