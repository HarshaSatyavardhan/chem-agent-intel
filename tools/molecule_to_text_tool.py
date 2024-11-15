# tools/molecule_to_text_tool_optimized.py

from optimum.intel import OVModelForSeq2SeqLM
from crewai_tools import BaseTool
from transformers import T5Tokenizer

import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams

from pydantic import Field

class MoleculeToTextTool(BaseTool):
    name: str = Field(default="Molecule to Text Converter")
    description: str = Field(
        default="Converts SMILES molecular representations to text descriptions using MolT5 with OpenVINO acceleration"
    )

    def __init__(self, model_id="laituan245/molt5-large-smiles2caption", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id, model_max_length=512)
        
        # Configure OpenVINO optimization
        ov_config = {
            hints.performance_mode(): hints.PerformanceMode.LATENCY,
            streams.num(): "1",
            props.cache_dir(): "",
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",  # CPU optimization
            "KV_CACHE_PRECISION": "u8",  # Memory optimization
        }
        
        self.model = OVModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            export=True,
            ov_config=ov_config,
            device="CPU",  # Use CPU acceleration
            load_in_8bit=True  # Enable int8 quantization
        )

    def _run(self, smiles: str) -> str:
        try:
            # Tokenize input SMILES string
            inputs = self.tokenizer(smiles, return_tensors="pt")
            
            # Perform inference
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            text_description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return text_description
        except Exception as e:
            print(f"Error in molecule-to-text conversion: {str(e)}")
            return f"Error in molecule-to-text conversion: {str(e)}"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
