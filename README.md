# 🌾🚜 Smart Grocery Price Optimization & APMC Logistics (OpenEnv)

![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Built for Agents](https://img.shields.io/badge/AI-Agent_Ready-orange)

A highly realistic, real-world OpenEnv simulation environment designed to train and evaluate AI agents on **agricultural supply chain logistics**, **spatial-temporal arbitrage**, and **inventory quality management**. 

---

## 🌎 1. The Real-World Problem (Motivation)
Globally, over **30% of agricultural produce is wasted** due to supply chain inefficiencies. Smallholder farmers rely on APMC (Agricultural Produce Market Committee) systems and wholesale markets where prices fluctuate wildly based on undocumented, hyper-local demand. 

Farmers often face a complex temporal and spatial optimization problem:
*   *Should I sell locally today for $400/ton?*
*   *Should I pay $5/ton/day to put it in cold-storage and wait for a festival price spike?*
*   *Should I pay $50/ton to rent a truck and ship it to a city market 2 days away, hoping prices don't crash by the time it arrives?*

This environment provides a **Constraint-Satisfaction Logistics Simulator** for an AI Agent to act as a "Smart Farmer Copilot." It must parse weather predictions, balance capital vs. spoilage, manage transit delays, and respect strict hard-capped market capacities.

---

## 🛠️ 2. The Agentic Interface (OpenEnv Spec)

The simulation rigorously follows the OpenEnv API requirements leveraging strongly typed `Pydantic` models for safety and determinism.

### 🧠 The Observation Space
Agents receive a complex `Observation` JSON on every step representing the real-world state:
*   `current_day` & `current_weather` (Normal, Rain, Heat)
*   `current_inventory` & `current_balance` (Operating Capital)
*   `crop_quality` (A fractional multiplier tied to spoilage)
*   `in_transit_inventory` (Tons locked on trucks)
*   `known_market_data` & `known_logistics_data` (Data built by the agent querying the environment)
*   `recent_events` (Crucial textual alerts like arrival confirmations or weather updates!)

### ⚙️ The Action Space (With Explainability!)
To push LLM agents to show true reasoning (and prevent hallucinated random clicks), every action requires a `reasoning` string explaining the strategic intent.

*   `query_market`: Retrieves current price and total demand for a market.
*   `query_logistics`: Retrieves trucking costs and transit delay days.
*   `predict_price`: Pay a monetary fee to unlock a probabilistic price forecast for future days.
*   `wait`: Hold inventory on the farm (Zero fee, but **high spoilage**).
*   `store_crop`: Hold inventory in cold-storage (High daily cash fee, but **low spoilage**).
*   `transport_crop`: Allocate inventory to a destination. (Locks capital and inventory in transit).
*   `sell_crop`: Execute a direct sale at a market for immediate revenue.

---

## 📉 3. The Simulator State & "Judge-Killer" Mechanics

The internal `simulator.py` engine is designed specifically to prevent generic LLMs from brute-forcing the environment:
1.  **Weather Engine:** Stochastic weather events dynamically alter the state. *Rain* delays all in-transit shipments by +1 day. *Heat* accelerates on-farm crop spoilage significantly.
2.  **Hard Capacity Rejects:** Markets have strict daily demand caps. If an agent naively tries to dump 100 tons into a 20-ton market, the environment hits it with a heavy **normalized cash penalty** (`-50.0 / 1000.0`) and rejects the sale.
3.  **Strict Normalization:** Reward outputs are divided by `1000.0` to keep continuous signals stable and prevent Q-value explosion.
4.  **Anti-Memorization Randomization:** Base prices, distance costs, maximum demands, and "festival surge" days are randomized on `reset()`, forcing the agent to actually "play" the environment instead of hardcoding a solution.

---

## 🏆 4. The 3 Grading Scenarios (Tasks)

The environment grades the agent continuously on a `0.0` to `1.0` scale by calculating a "Theoretical Maximum Profit" given the randomized parameters initialized at `reset()`. 

1.  🟢 **`easy_arbitrage` (Day Trading):**
    *   **Goal:** Optimize profit across 2 local markets in a single day.
    *   **Focus:** Basic JSON parsing, logic gating, comparing two integers, and correctly utilizing `query_market`.
2.  🟡 **`medium_temporal` (Time-Series Optimization):**
    *   **Goal:** The agent must manage 100 tons of crop over 5 days. A massive price spike will occur on Day 4. 
    *   **Focus:** Trade-off analysis. The agent must calculate if it is mathematically superior to eat heavy spoilage (`wait`) or bleed operational cash (`store_crop`) to maximize the multiplier when the spike hits.
3.  🔴 **`hard_logistics` (Multi-Market Supply Chain):**
    *   **Goal:** 10 days, 100 tons, 1 local market, and 2 distant city markets.
    *   **Focus:** Complex logistics. The agent must query transport costs, factor in 2-3 day transit delays, accurately predict prices at arrival, and split their shipments perfectly so they do not exceed the narrow daily capacity caps at the destination cities.

---

## 🚀 5. How to Run & Verify

This environment is fully prepped for **Hugging Face Spaces**. 
The repository includes a ready-to-run `inference.py` script that hooks the OpenAI python client directly into the OpenEnv `APMCEnv` integration.

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your Credentials**
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini" # Or any capable model
```

**3. Run the Baseline Assessment**
```bash
python inference.py
```

The script will initiate all 3 tasks, print the agent's observation/action JSON exchanges, handle any OpenEnv interactions seamlessly, and deliver a final normalized metric-grade!
