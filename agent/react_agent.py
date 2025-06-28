# react_agent.py
import json
import logging
import re
import os
from tools.tool_list import tool_list
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

# === Load LLM ===
# model_name = "deepseek-ai/deepseek-llm-7b-chat" #7B Params
# model_name = "mistralai/Mistral-7B-Instruct-v0.3" #7B Params
# model_name = "microsoft/Phi-3-mini-4k-instruct" #3.8B Params
model_name = "microsoft/phi-2" #Smaller model

use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else -1

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",torch_dtype=torch.float16)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Prompt Templates ===
REACT_TEMPLATE = """
You are a ReAct agent that reasons and takes actions using tools.
Use the following format:

Thought: think about the problem
Action: {{ "tool": "tool_name", "args": {{...}} }}
Observation: result of the action
... (repeat as needed) ...
Final Answer: your final reply to the user

Available tools:
{tool_list}

**Important Notes:**
- Use ONLY the tools listed below.
- Only call tools if they are needed to answer the question.
- Do not call tools just for the sake of it.
- If unsure, think step by step and end with Final Answer.

User Query: {query}
{memory}
Begin your reasoning below:
"""

CLARIFICATION_PROMPT = """
You are a helpful assistant. The user's query is vague or incomplete.
Kindly ask them a clarifying question to get more context.

User Query: "{query}"
Assistant:
"""

NEXT_STEP_PROMPT = """
You just answered the user's query:

User Query: "{user_query}"

Answer: "{answer}"

Based on the user's original question and your answer, suggest 3 short and helpful follow-up questions or actions the user might want to take next.

Be concise and focused. Use this format:
1. ...
2. ...
3. ...

Avoid repeating information or inventing elaborate scenarios.
"""


BASIC_PROMPT = """
You are a helpful assistant. Based on the following user query and the retrieved information, provide a clear and helpful answer.
User Query: {user_query}

Information Retrieved:
{observation}

Now, generate a helpful answer for the user. Start your answer with the word "Answer:"
"""

# === Helpers  ===
def format_tools():
    return "\n".join(
        f"- {name}: {getattr(tool, 'description', 'No description')}"
        for name, tool in tool_list.items()
    )

def extract_final_answer(llm_output: str) -> str:
    match = re.search(r"Answer:\s*(.*)", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_output.strip()

def extract_actions(text):
    return re.findall(r'Action:\s*(\{.*?\})', text, re.DOTALL)

def clean_observations(obs, max_length=2000):
    if isinstance(obs, dict):
        cleaned = []
        for _, chunks in obs.items():
            if isinstance(chunks, list):
                cleaned.extend(chunks)

        joined = " ".join(c.strip().replace("\n", " ") for c in cleaned)

        # Truncate to avoid hitting LLM max length
        return joined[:max_length] + "..." if len(joined) > max_length else joined

    elif isinstance(obs, str):
        return obs.replace("\n", " ").strip()[:max_length]
    
    return str(obs)[:max_length]


def llm_chat(prompt, max_tokens=512):
    output = generator(prompt, max_new_tokens=max_tokens, temperature=0.7)
    print(f"[llm_chat] Output received. {output}")
    return output[0]['generated_text'][len(prompt):].strip()

def pass_llm_query(route_result, user_roles=None):
    user_query = route_result["raw_input"]
    tool_name = route_result.get("intent")
    observation = ""

    tool = tool_list.get(tool_name)
    if not tool:
        return f"❌ Unknown tool '{tool_name}'"

    try:
        if hasattr(tool, "search_grouped"):
            observation = tool.search_grouped(query=user_query, user_roles=user_roles)
        else:
            observation = tool(query=user_query)
    except Exception as e:
        logger.debug(f"❌ Exception while running tool: {e}")
        return f"❌ Tool '{tool_name}' failed to run: {str(e)}"

    if isinstance(observation, dict) and "error" in observation:
        error_message = "\n".join(observation["error"])
        logger.debug("[Tool] ❌ Error detected in observation.")
        return f"⚠️ Tool Error:\n{error_message}"
    clean_observation = clean_observations(observation, max_length=2000) #Prevent it from excedding the model Max Length
    prompt = BASIC_PROMPT.format(user_query=user_query, observation=clean_observation)
    logger.debug(f"[LLM] prompt: {prompt}")

    answer = llm_chat(prompt)
    clean_answer = extract_final_answer(answer)
    print(F"[LLM] Clean Answer : {clean_answer}")
    next_prompt = NEXT_STEP_PROMPT.format(user_query=user_query, answer=clean_answer)
    next_steps = llm_chat(next_prompt, max_tokens=128)

    return f"{clean_answer}\n\n🤖 What's Next?\n{next_steps.strip()}"


# === Clarification
def clarify_query(user_query: str) -> str:
    logger.debug(f"Seeking Clarification: {user_query}")
    prompt = CLARIFICATION_PROMPT.format(query=user_query)
    response = llm_chat(prompt)
    return response.split("Assistant:")[-1].strip()