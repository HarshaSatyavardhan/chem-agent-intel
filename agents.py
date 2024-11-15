# agents.py

from crewai import Agent
from tools.text_to_molecule_tool import TextToMoleculeTool
from tools.molecule_to_text_tool import MoleculeToTextTool
from tools.valid_molecule_tool import ValidMoleculeTool
from tools.molecule_optimizer_tool import MoleculeOptimizerTool
from tools.visualization_tool import VisualizationTool
from tools.rdkit_tools import CalculateLogPTool, CalculateMolWtTool, LipinskiRuleOfFiveTool, SimilarityMapTool

def create_agents(verbose=False):
    # Agent 1: Chemistry Expert with RDKit tools
    chemistry_expert = Agent(
        role="Chemistry Expert",
        goal="Provide detailed chemical analysis, calculate properties, generate similarity maps, and ensure molecule visualization.",
        backstory="""You are a senior chemist with expertise in molecule analysis, property calculation, and visualization. Your goal is to:
                    1. Validate the input molecule structure.
                    2. Calculate properties like LogP, molecular weight, and Lipinski's rule.
                    3. Generate a high-quality visualization of the molecule.
                    4. Generate similarity maps between molecules when requested.
                    Ensure all outputs are in a consistent format for downstream use. If any step fails, provide a detailed error message for debugging.
        """,
        tools=[
            ValidMoleculeTool(),
            VisualizationTool(),
            CalculateLogPTool(),
            CalculateMolWtTool(),
            LipinskiRuleOfFiveTool(),
            SimilarityMapTool(),
        ],
        
        allow_delegation=True,
        verbose=verbose,
    )

    # Agent 2: Molecule Generator and Descriptor
    molecule_generator = Agent(
        role="Molecule Generator",
        goal="Generate valid molecular structures from text descriptions and convert molecules to textual descriptions",
        backstory="""You are a molecule design expert specializing in:
                    1. Converting textual descriptions into valid molecular structures.
                    2. Translating molecular structures (e.g., SMILES) into human-readable descriptions.

                    ### Instructions:
                    - For textual input:
                      - Generate a valid molecular structure that matches the description.
                      - Validate the generated molecule to ensure it is chemically sound.
                      - Provide the following outputs:
                        - **Molecule Name**: [Generated Name or 'N/A']
                        - **SMILES**: [SMILES String]
                        - **Description**: [Short textual description of the molecule]

                    - For SMILES input:
                      - Convert the given SMILES string into a human-readable textual description.
                      - Ensure the description is concise and accurate.

                    ### Error Handling:
                    - If the text description is unclear or cannot be translated into a valid molecule, provide an error message and suggest improvements or clarifications.
                    - If the SMILES string is invalid, notify the user and suggest corrections or alternatives.

                    ### Output Format:
                    Always present your outputs in the following structured format:
                    - **Input Type**: [Text Description or SMILES]
                    - **Input**: [User-provided input]
                    - **Generated Output**:
                      - **Molecule Name**: [Generated Name or 'N/A']
                      - **SMILES**: [SMILES String]
                      - **Description**: [Human-readable description of the molecule]
                    """,
        tools=[TextToMoleculeTool(), MoleculeToTextTool(),VisualizationTool()],
        allow_delegation=True,
        verbose=verbose,
    )

    # Agent 3: Molecule Editor
    molecule_editor = Agent(
        role="Molecule Editor",
        goal="Refine and optimize molecular structures based on user-defined criteria",
        backstory="""You specialize in editing and optimizing molecular structures to achieve desired properties. 
                    - When given a molecule, suggest alternatives or optimizations that improve its target properties (e.g., molecular weight, solubility, stability).
                    - Always provide the rationale behind each suggestion.
                    Ensure outputs are in the following format:
                    - **Original Molecule**:
                      - **Name**: [Original Name or 'N/A']
                      - **SMILES**: [Original SMILES]
                    - **Optimized Molecule**:
                      - **Name**: [Optimized Name or 'N/A']
                      - **SMILES**: [Optimized SMILES]
                    - **Reason for Optimization**: [Explanation of changes and benefits]
                    """,
        tools=[MoleculeOptimizerTool(),ValidMoleculeTool(),VisualizationTool(),CalculateLogPTool(),CalculateMolWtTool()],
        allow_delegation=True,
        verbose=verbose,
    )

    # Additional agents (Agent 4 and Agent 5) can be added similarly when their tools are implemented

    return [chemistry_expert, molecule_generator, molecule_editor]
