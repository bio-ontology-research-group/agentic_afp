from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool
# from dotenv import load_dotenv
import os

from src.uniprot.search_uniprot import search_uniprot

# load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENROUTER,
    model_type="google/gemini-2.0-flash-001",
    api_key=os.environ.get(OPENROUTER_API_KEY),
    model_config_dict={"temperature": 0.3, "max_tokens": 1000000},
    
)

search_uniprot_tool = FunctionTool(search_uniprot)

def test_uniprot_agent():
    context = "You are a helpful assistant that can search UniProt for protein information. You are given a protein sequence and a hypothesis about its function. The function search_uniprot will look for similarproteins in UniProt and return their GO annotations. Use this information to determine if the hypothesis is supported by the data."

    sequence = "MPYKLKKEKEPPKVAKCTAKPSSSGKDGGGENTEEAQPQPQPQPQPQAQSQPPSSNKRPSNSTPPPTQLSKIKYSGGPQIVKKERRQSSSRFNLSKNRELQKLPALKDSPTQEREELFIQKLRQCCVLFDFVSDPLSDLKFKEVKRAGLNEMVEYITHSRDVVTEAIYPEAVTMFSVNLFRTLPPSSNPTGAEFDPEEDEPTLEAAWPHLQLVYEFFLRFLESPDFQPNIAKKYIDQKFVLALLDLFDSEDPRERDFLKTILHRIYGKFLGLRAYIRRQINHIFYRFIYETEHHNGIAELLEILGSIINGFALPLKEEHKMFLIRVLLPLHKVKSLSVYHPQLAYCVVQFLEKESSLTEPVIVGLLKFWPKTHSPKEVMFLNELEEILDVIEPSEFSKVMEPLFRQLAKCVSSPHFQVAERALYYWNNEYIMSLISDNAARVLPIMFPALYRNSKSHWNKTIHGLIYNALKLFMEMNQKLFDDCTQQYKAEKQKGRFRMKEREEMWQKIEELARLNPQYPMFRAPPPLPPVYSMETETPTAEDIQLLKRTVETEAVQMLKDIKKEKVLLRRKSELPQDVYTIKALEAHKRAEEFLTASQEAL"

    hypothesized_function_1 = "GO:0110165"
    hypothesized_function_2 = "GO:0000000"

    uniprot_agent = ChatAgent(context, tools=[search_uniprot_tool], model=model)

    print("Testing with hypothesized function:", hypothesized_function_1)
    prompt = f"Analyze the following protein sequence and determine if it supports the hypothesized function {hypothesized_function_1}:\n\n{sequence}"
    response = uniprot_agent.step(prompt)
    interpretation = response.msgs[0].content
    print(f"Agent's interpretation: {interpretation}")

    uniprot_agent.reset()
    print("\nTesting with hypothesized function:", hypothesized_function_2)
    prompt = f"Analyze the following protein sequence and determine if it supports the hypothesized function {hypothesized_function_2}:\n\n{sequence}"
    response = uniprot_agent.step(prompt)
    interpretation = response.msgs[0].content
    print(f"Agent's interpretation: {interpretation}")
    
if __name__ == "__main__":
    test_uniprot_agent()
