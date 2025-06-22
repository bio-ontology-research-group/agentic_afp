import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
max_tokens = 140000

deepseek_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENROUTER,
    model_type="deepseek/deepseek-chat-v3-0324:free",
    api_key=OPENROUTER_API_KEY,
    model_config_dict={"temperature": 0.3, "max_tokens": 140000},
)

# model = ModelFactory.create(
    # model_platform=ModelPlatformType.OPENROUTER,
    # model_type="deepseek/deepseek-chat-v3-0324:free",
    # model_type="meta-llama/llama-3.3-8b-instruct:free",
    # model_type="google/gemini-2.0-flash-001",
    # api_key=OPENROUTER_API_KEY,
    # model_config_dict={"temperature": 0.3, "max_tokens": 140000},
# )

