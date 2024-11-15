# tools/molecule_to_text_tool.py

from crewai_tools import BaseTool
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pydantic import Field
import torch

class MoleculeToTextTool(BaseTool):
    name: str = Field(default="Molecule to Text Converter")
    description: str = Field(
        default="Converts SMILES representations to text descriptions using MolT5 model"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained(
            "laituan245/molt5-large-smiles2caption", model_max_length=512
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            "laituan245/molt5-large-smiles2caption"
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def _run(self, smiles: str) -> str:
        try:
            input_ids = self.tokenizer(smiles, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            outputs = self.model.generate(
                input_ids, num_beams=5, max_length=512, temperature=0.7, do_sample=True
            )
            description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return description
        except Exception as e:
            print(f"Error in molecule to text conversion: {str(e)}")
            return f"Error in molecule to text conversion: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
