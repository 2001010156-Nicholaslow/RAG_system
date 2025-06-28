from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_name = "deepseek-ai/deepseek-llm-7b-chat" #7B Params
# model_name = "mistralai/Mistral-7B-Instruct-v0.3" #7B Params
# model_name = "microsoft/Phi-3-mini-4k-instruct" #3.8B Params
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "You are a helpful assistant. What is the capital of France?"
output = generator(prompt, max_new_tokens=50)
print(output)