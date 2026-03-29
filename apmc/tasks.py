import random
from .models import State, TransportCost

def generate_easy_task() -> tuple[State, float]:
    price_a = random.randint(400, 500)
    price_b = random.randint(550, 650)
    demand_a = random.randint(80, 120)
    demand_b = random.randint(30, 120)

    state = State(
        current_day=1,
        max_days=1,
        current_balance=0.0,
        current_inventory=50.0,
        crop_quality=1.0,
        true_market_prices={1: {"Local_A": price_a, "Local_B": price_b}},
        true_market_demand={1: {"Local_A": demand_a, "Local_B": demand_b}},
        logistics_map={
            "Local_A": TransportCost(cost_per_ton=0.0, days_to_arrive=0),
            "Local_B": TransportCost(cost_per_ton=0.0, days_to_arrive=0)
        },
        in_transit=[],
        inventory_at_markets={},
        sold_history=[]
    )
    
    # Calculate perfect score theoretically
    best = "Local_B" if price_b > price_a else "Local_A"
    best_price = max(price_a, price_b)
    best_demand = demand_b if best == "Local_B" else demand_a
    qty_sold = min(50.0, best_demand)
    max_profit = qty_sold * best_price
    
    # If the highest price market couldn't buy it all, we'd sell the rest to the other market
    if qty_sold < 50.0:
         leftover = 50.0 - qty_sold
         other_price = price_a if best == "Local_B" else price_b
         other_demand = demand_a if best == "Local_B" else demand_b
         max_profit += min(leftover, other_demand) * other_price

    return state, max_profit

def generate_medium_task() -> tuple[State, float]:
    base_price = random.randint(350, 450)
    spike_price = random.randint(750, 950)
    
    prices = {
        1: {"FestivalMarket": base_price},
        2: {"FestivalMarket": base_price},
        3: {"FestivalMarket": base_price},
        4: {"FestivalMarket": spike_price}, # Random Festival spike!
        5: {"FestivalMarket": base_price},
    }
    demand = {d: {"FestivalMarket": random.randint(200, 500)} for d in range(1, 6)}
    
    state = State(
        current_day=1,
        max_days=5,
        current_balance=0.0,
        current_inventory=100.0,
        crop_quality=1.0,
        storage_cost_per_ton_day=5.0,
        spoilage_rate_waiting=0.10,
        spoilage_rate_stored=0.01,
        true_market_prices=prices,
        true_market_demand=demand,
        logistics_map={
            "FestivalMarket": TransportCost(cost_per_ton=0.0, days_to_arrive=0)
        },
        in_transit=[],
        inventory_at_markets={},
        sold_history=[]
    )
    # Optimum: Store for 3 days then sell
    storage_cost = 100.0 * 5.0 * 3.0
    final_quality = 1.0 - (0.01 * 3)
    revenue = 100.0 * spike_price * final_quality
    max_profit = revenue - storage_cost
    return state, max_profit

def generate_hard_task() -> tuple[State, float]:
    prices = {}
    demand = {}
    
    # Define random market economics
    city_a_base = random.randint(450, 550)
    city_b_base = random.randint(550, 650)
    local_base = random.randint(150, 250)
    
    # Add varying random events over 10 days
    for d in range(1, 11):
        # Allow +/- 10% daily variation
        prices[d] = {
            "LocalMarket": max(10, local_base + random.randint(-20, 20)), 
            "CityMarket_A": city_a_base + random.randint(-50, 50), 
            "CityMarket_B": city_b_base + random.randint(-50, 50)
        }
        # Hard limits on long range demand to enforce split shipping
        demand[d] = {
            "LocalMarket": 9999.0, 
            "CityMarket_A": random.randint(25, 40), 
            "CityMarket_B": random.randint(15, 30)
        }
        
    state = State(
        current_day=1,
        max_days=10,
        current_balance=0.0,
        current_inventory=100.0,
        crop_quality=1.0,
        true_market_prices=prices,
        true_market_demand=demand,
        logistics_map={
            "LocalMarket": TransportCost(cost_per_ton=0.0, days_to_arrive=0),
            "CityMarket_A": TransportCost(cost_per_ton=50.0, days_to_arrive=2),
            "CityMarket_B": TransportCost(cost_per_ton=100.0, days_to_arrive=3)
        },
        in_transit=[],
        inventory_at_markets={},
        sold_history=[]
    )
    
    # Estimate an upper bound for scoring (sending perfectly to city markets without delay)
    best_net_a = city_a_base - 50
    best_net_b = city_b_base - 100
    avg_best = (best_net_a + best_net_b) / 2
    max_profit = 100.0 * avg_best * 0.95 # Assumes 5% loss to spoilage/delays
    
    return state, max_profit

TASKS = {
    "easy_arbitrage": generate_easy_task,
    "medium_temporal": generate_medium_task,
    "hard_logistics": generate_hard_task
}
