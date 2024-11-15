# Molecule Design Assistant 

![Main](images/main.png)


> [Disclaimer]

> The **Project** folder contains the same app code where the agents and tools are not openvino optimized

- tools folder contains code 

- To run the app follow the instructions 

- clone this repo
- cd `chem-agent-intel`
- install the virtual environment 

`pip install virtualenv`


`virtualenv -p python3.10 intel`

- activate the env

`source intel/bin/activate`


install all these packages in the environment 

```
pip install transformers 
pip install -U langchain-community
pip install rdkit-pypi 
pip install gradio
pip install langchain 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentencepiece
pip install -U langchain-community
pip install streamlit-shadcn-ui
pip install openvino-dev optimum-intel
pip install optimum[openvino,nncf] torchvision evaluate
pip install crewai
```

- In the `main.py` file add your own `OPENAI API KEY`

then run streamlit app using the command 

- `streamlit run main.py`


## showing the usage of the Intel openvino and AMX

![slides](images/slides.png)
![openvino](images/openvino.png)
![amx](images/amx.png)


- Benchmarks comapring the inference speed once optimized using openvino and without it is shown in this `inference_speed_openvino.ipynb` notebook and benchmarked 

- In `run.ipynb` notebook solubility prediction tool is implemented and show a way to use ngrok to host the streamlit into a temporary domain need to add your own 


## Prompts for testing the app

```
Prompt 1:

Generate a molecule for this description and visualize the molecule : The molecule is a monomethoxybenzene that is 2-methoxyphenol substituted by a hydroxymethyl group at position 4. It has a role as a plant metabolite. It is a member of guaiacols and a member of benzyl alcohols.


Prompt 2: 

{generated molecule}
[N][#C][C][C][C@@H1][C][C][C][C][C][C][C][C][C][C][Ring1][N][=O] optimize this molecule

Prompt 3:

COc1ccc(cc1C=O)O generate description for this molecule 

Prompt 4:

As a Chemistry Expert, generate the similarity map of both the molecules get the similarity map of both the molecules: O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C and  OC(=O)C[C@H](O)C[C@H](O)\C=C\c1c(C(C)C)nc(N(C)S(=O)(=O)C)nc1c2ccc(F)cc2


Prompt 5:

generate the similarity map of both the molecules get the similarity map of both the molecules: O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C and  OC(=O)C[C@H](O)C[C@H](O)\C=C\c1c(C(C)C)nc(N(C)S(=O)(=O)C)nc1c2ccc(F)cc2

```



## Using CrewAI for multi agent conversation in this project 
![crew](images/crew.png)

