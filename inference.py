import os
import json
import re
from openai import OpenAI
from apmc.env import APMCEnv
from apmc.tasks import TASKS
from apmc.models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

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
        print("Failed to parse action JSON:", response_text)
        return {"reasoning": "Fallback", "action_type": "wait", "days": 1}

def run_task(task_name: str, client: OpenAI) -> float:
    print(f"\n========== RUNNING TASK: {task_name} ==========")
    env = APMCEnv(task_name=task_name)
    obs = env.reset()
    done = False
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for step in range(30): 
        obs_json = obs.model_dump_json(indent=2)
        print(f"\n--- STEP {step+1} ---")
        print("Observation:")
        print(obs_json)
        
        messages.append({"role": "user", "content": f"Observation:\n{obs_json}\nWhat is your next action JSON?"})
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0
            )
            reply = response.choices[0].message.content or ""
            print("\nAgent Reply:", reply)
        except Exception as e:
            print("OpenAI API Error:", e)
            return 0.0

        messages.append({"role": "assistant", "content": reply})
        
        action_dict = extract_action(reply)
        try:
            action = Action(**action_dict)
            print(f"Reasoning: {action.reasoning}")
        except Exception as e:
            print("Invalid Action structure:", e)
            action = Action(reasoning="Recovery from error", action_type="wait", days=1)
            
        print(f"Executed: {action.action_type}")
        obs, reward, done, info = env.step(action)
        print(f"Immediate Normalized Reward: {reward.value}")
        
        if done:
            print(f"\n[!] Episode Completed. Task: {task_name}")
            print(f"[+] Final Grade: {info.grade}")
            print(f"[+] Final Metrics: {info.metrics}")
            return info.grade
            
    print("\n[!] Error: Exceeded max steps.")
    return 0.0

def main():
    if not API_KEY:
        print("WARNING: HF_TOKEN or OPENAI_API_KEY environment variable not set. LLM calls will fail.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    scores = {}
    for task_name in TASKS.keys():
        score = run_task(task_name, client)
        scores[task_name] = score
        print("\n" + "="*50)
        
    print("\n========== FINAL RESULTS ==========")
    for k, v in scores.items():
        print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()
