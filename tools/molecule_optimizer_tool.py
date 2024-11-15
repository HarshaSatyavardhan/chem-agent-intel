# tools/molecule_optimizer_tool.py

from crewai_tools import BaseTool
from transformers import AutoTokenizer
from pydantic import Field
from rdkit import Chem

from optimum.intel import OVModelForSeq2SeqLM

import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams


class MoleculeOptimizerTool(BaseTool):
    name: str = Field(default="Molecule Optimizer")
    description: str = Field(
        default="Suggests optimizations for molecular properties using MolGen model with OpenVINO acceleration"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large-opt")

        ov_config = {
            hints.performance_mode(): hints.PerformanceMode.LATENCY,
            streams.num(): "1",
            props.cache_dir(): "",
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
            "KV_CACHE_PRECISION": "u8",
        }

        self.model = OVModelForSeq2SeqLM.from_pretrained(
            "zjunlp/MolGen-large-opt",
            export=True,
            ov_config=ov_config,
            device="CPU",
            load_in_8bit=True,
        )

    def _run(self, smiles: str) -> list:
        try:
            sf_input = self.tokenizer(smiles, return_tensors="pt")

            molecules = self.model.generate(
                input_ids=sf_input["input_ids"],
                max_length=512,
                num_beams=10,
                num_return_sequences=5,
                temperature=0.7,
                do_sample=True,
            )

            suggestions = [
                self.tokenizer.decode(mol, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                .replace(" ", "").strip()
                for mol in molecules
                if Chem.MolFromSmiles(
                    self.tokenizer.decode(mol, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    .replace(" ", "").strip()
                )
                is not None
            ]

            return suggestions if suggestions else ["No valid optimizations found"]
        except Exception as e:
            return [f"Error in optimization: {str(e)}"]

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
