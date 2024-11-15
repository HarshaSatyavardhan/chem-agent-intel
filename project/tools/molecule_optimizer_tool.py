# tools/molecule_optimizer_tool.py

from crewai_tools import BaseTool
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import Field
from rdkit import Chem
import torch

class MoleculeOptimizerTool(BaseTool):
    name: str = Field(default="Molecule Optimizer")
    description: str = Field(
        default="Suggests optimizations for molecular properties using MolGen model"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large-opt")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("zjunlp/MolGen-large-opt")

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def _run(self, smiles: str) -> list:
        try:
            sf_input = self.tokenizer(smiles, return_tensors="pt")
            if torch.cuda.is_available():
                sf_input = {k: v.to("cuda") for k, v in sf_input.items()}

            molecules = self.model.generate(
                input_ids=sf_input["input_ids"],
                num_beams=10,
                num_return_sequences=5,
                max_length=512,
                temperature=0.7,
                do_sample=True,
            )

            suggestions = []
            for mol in molecules:
                smiles_output = (
                    self.tokenizer.decode(
                        mol, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    .replace(" ", "")
                    .strip()
                )
                if Chem.MolFromSmiles(smiles_output) is not None:
                    suggestions.append(smiles_output)

            if not suggestions:
                suggestions = ["No valid optimizations found"]
            
            
            print(f"Optimized Molecule Suggestions: {suggestions}")

            return suggestions
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            return [f"Error in optimization: {str(e)}"]

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
