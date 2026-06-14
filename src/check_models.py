from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Models to test
models_to_test = [
    ("together", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("together", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    ("together", "deepseek-ai/DeepSeek-R1"),
    ("nebius", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("featherless-ai", "meta-llama/Meta-Llama-3-8B-Instruct"),
]

for provider, model in models_to_test:
    try:
        client = InferenceClient(
            provider=provider,
            api_key=HF_TOKEN
        )
        response = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
            temperature=0.1
        )
        print(f"✅ WORKS → provider={provider} | model={model}")
        print(f"   Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ FAILED → provider={provider} | model={model}")
        print(f"   Error: {str(e)[:100]}")
    print("-" * 60)