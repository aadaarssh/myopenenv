import os
import json
import re
from openai import OpenAI
from apmc.env import APMCEnv
from apmc.tasks import TASKS
from apmc.models import Action

# Pre-Submission Checklist explicitly requires exact os.getenv setup for these three variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are a smart agricultural supply chain agent.
Your objective is to maximize profit by deciding where and when to sell crops.
You will receive observations about current inventory, balance, market data, and logistics.
You must respond with EXACTLY ONE action formatted as a JSON object.
Allowed actions and parameters:
{"reasoning": "...", "action_type": "query_market", "market_id": "..."}
{"reasoning": "...", "action_type": "query_logistics", "market_id": "..."}
{"reasoning": "...", "action_type": "predict_price", "market_id": "...", "days": X}
{"reasoning": "...", "action_type": "wait", "days": X}
{"reasoning": "...", "action_type": "store_crop", "days": X}
{"reasoning": "...", "action_type": "transport_crop", "market_id": "...", "quantity": X}
{"reasoning": "...", "action_type": "sell_crop", "market_id": "...", "quantity": X}

Always provide your tactical reasoning.
Always output only raw JSON representing the action. Do not wrap it in markdown block quotes.
"""

def extract_action(response_text: str) -> dict:
    try:
        match = re.search(r'\{.*\}', response_text.replace('\n', ''))
        if match:
            return json.loads(match.group(0))
        return json.loads(response_text)
    except Exception as e:
        return {"reasoning": "Fallback", "action_type": "wait", "days": 1}

def run_task(task_name: str, client: OpenAI) -> float:
    # Must follow exact (START/STEP/END) formatting rule
    print(f"START {task_name}")
    env = APMCEnv(task_name=task_name)
    obs = env.reset()
    done = False
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for step in range(30): 
        print(f"STEP {step+1}")
        obs_json = obs.model_dump_json(indent=2)
        
        messages.append({"role": "user", "content": f"Observation:\n{obs_json}\nWhat is your next action JSON?"})
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            print("OpenAI API Error:", e)
            print(f"END {task_name} 0.0")
            return 0.0

        messages.append({"role": "assistant", "content": reply})
        
        action_dict = extract_action(reply)
        try:
            action = Action(**action_dict)
        except Exception as e:
            action = Action(reasoning="Recovery from error", action_type="wait", days=1)
            
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"END {task_name} {info.grade}")
            return info.grade
            
    print(f"END {task_name} 0.0")
    return 0.0

def main():
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: HF_TOKEN or OPENAI_API_KEY environment variable not set. LLM calls will fail.")
        
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    
    scores = {}
    for task_name in TASKS.keys():
        score = run_task(task_name, client)
        scores[task_name] = score

if __name__ == "__main__":
    main()
