from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apmc.env import APMCEnv
from apmc.models import Action

app = FastAPI()
env_instance = None

class ResetRequest(BaseModel):
    task_name: str = "easy_arbitrage"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Smart Grocery APMC Logistics OpenEnv API"}

@app.post("/reset")
def reset(req: ResetRequest = None):
    global env_instance
    task = req.task_name if req else "easy_arbitrage"
    env_instance = APMCEnv(task_name=task)
    obs = env_instance.reset()
    return {"observation": obs.model_dump()}

@app.post("/step")
def step(action: Action):
    global env_instance
    if not env_instance:
        raise HTTPException(status_code=400, detail="Must call /reset before /step")
    
    obs, reward, done, info = env_instance.step(action)
    return {
         "observation": obs.model_dump(),
         "reward": reward.model_dump(),
         "done": done,
         "info": info.model_dump()
    }

@app.get("/state")
def get_state():
    global env_instance
    if not env_instance:
        raise HTTPException(status_code=400, detail="Must call /reset first")
    return {"state": env_instance.state()}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
