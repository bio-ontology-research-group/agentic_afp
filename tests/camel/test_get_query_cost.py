from unittest import TestCase
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
import os
from src.utils import get_query_cost

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class TestGetQueryCost(TestCase):

    def setUp(self):
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENROUTER,
            model_type="deepseek/deepseek-chat-v3-0324:free",
            api_key=OPENROUTER_API_KEY,
            model_config_dict={"temperature": 0.3, "max_tokens": 100000},
        )


        context = "You are a simple agent used for a test to get query cost."
        
        self.agent = ChatAgent(context, model=model)
        
    def test_get_generation_id(self):

        prompt = "What is the results of 3+4?"

        response = self.agent.step(prompt)

        print(f"Response: {response}")
        gen_id = response.info['id']
        self.assertTrue(gen_id.startswith("gen-"))
        


    def test_get_query_cost(self):
        prompt = "What is the results of 3+4?"
        
        response = self.agent.step(prompt)
        
        print(f"Response: {response}")
        gen_id = response.info['id']
        cost = get_query_cost(gen_id, OPENROUTER_API_KEY)
        
        print(f"Query cost for generation {gen_id}: {cost}")
        self.assertGreaterOrEqual(cost, 0.0, "Query cost should be non-negative.")
