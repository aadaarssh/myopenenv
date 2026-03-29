import random
from typing import Tuple, List, Dict
from .models import State, Action, ActionType, MarketObservation, TransportCost, PredictionResult, Shipment, Observation

class APMCSimulator:
    def __init__(self, state: State):
        self.state = state
        self.known_markets: Dict[str, MarketObservation] = {}
        self.known_logistics: Dict[str, TransportCost] = {}
        self.predictions: Dict[str, List[PredictionResult]] = {}
        self.events: List[str] = ["Simulation started. Welcome to the Smart Grocery Price Optimization Agent."]

    def get_observation(self) -> Observation:
        in_transit_total = sum(s.quantity for s in self.state.in_transit)
        return Observation(
            current_day=self.state.current_day,
            current_weather=self.state.current_weather,
            current_balance=self.state.current_balance,
            current_inventory=self.state.current_inventory,
            crop_quality=self.state.crop_quality,
            known_market_data=self.known_markets.copy(),
            known_logistics_data=self.known_logistics.copy(),
            predictions=self.predictions.copy(),
            in_transit_inventory=in_transit_total,
            recent_events=self.events.copy()
        )

    def process_action(self, action: Action) -> float:
        """Processes the action, mutates state, and returns normalized reward."""
        reward = 0.0
        self.events.clear()

        if action.reasoning:
            self.events.append(f"Agent reasoning: {action.reasoning}")

        if self.state.current_day > self.state.max_days:
             self.events.append("FAILED: Simulation has already ended.")
             return 0.0

        if action.action_type == ActionType.QUERY_MARKET:
            market_id = action.market_id
            if market_id in self.state.true_market_prices.get(self.state.current_day, {}):
                price = self.state.true_market_prices[self.state.current_day][market_id]
                demand = self.state.true_market_demand[self.state.current_day][market_id]
                self.known_markets[market_id] = MarketObservation(price=price, demand=demand)
                self.events.append(f"Queried {market_id}: Price=${price:.2f}/ton, Demand={demand:.1f} tons.")
            else:
                self.events.append(f"Failed to query {market_id}: Unknown market for this day.")
                reward -= 1.0 / 1000.0

        elif action.action_type == ActionType.QUERY_LOGISTICS:
            market_id = action.market_id
            if market_id in self.state.logistics_map:
                cost = self.state.logistics_map[market_id]
                self.known_logistics[market_id] = cost
                self.events.append(f"Queried logistics for {market_id}: Cost=${cost.cost_per_ton:.2f}/ton, Time={cost.days_to_arrive} days.")
            else:
                self.events.append(f"Failed to query logistics for {market_id}.")
                reward -= 1.0 / 1000.0

        elif action.action_type == ActionType.PREDICT_PRICE:
            market_id = action.market_id
            days_ahead = action.days or 1
            target_day = self.state.current_day + days_ahead
            if target_day <= self.state.max_days and market_id in self.state.true_market_prices.get(target_day, {}):
                true_price = self.state.true_market_prices[target_day][market_id]
                noise = true_price * 0.1 * random.uniform(-1, 1)
                pred_min = max(0, true_price + noise - (true_price * 0.05))
                pred_max = true_price + noise + (true_price * 0.05)
                
                prediction_cost = 50.0 
                self.state.current_balance -= prediction_cost
                reward -= prediction_cost / 1000.0

                if market_id not in self.predictions:
                    self.predictions[market_id] = []
                self.predictions[market_id].append(PredictionResult(
                    expected_price_min=round(pred_min, 2),
                    expected_price_max=round(pred_max, 2),
                    predicted_for_day=target_day
                ))
                self.events.append(f"Purchased prediction for {market_id} on day {target_day} for ${prediction_cost}.")
            else:
                self.events.append(f"Failed prediction for {market_id} at day {target_day}.")
                reward -= 1.0 / 1000.0

        elif action.action_type == ActionType.TRANSPORT_CROP:
            market_id = action.market_id
            qty = action.quantity or 0.0
            if qty <= 0 or qty > self.state.current_inventory:
                self.events.append(f"Failed transport: Invalid quantity {qty}.")
                reward -= 10.0 / 1000.0
            elif market_id not in self.state.logistics_map:
                self.events.append(f"Failed transport: Unknown market {market_id}.")
                reward -= 10.0 / 1000.0
            else:
                logistics = self.state.logistics_map[market_id]
                transport_cost = logistics.cost_per_ton * qty
                self.state.current_balance -= transport_cost
                reward -= transport_cost / 1000.0
                self.state.current_inventory -= qty
                
                arrival_day = self.state.current_day + logistics.days_to_arrive
                self.state.in_transit.append(Shipment(
                    market_id=market_id,
                    quantity=qty,
                    arrival_day=arrival_day,
                    quality_on_arrival=self.state.crop_quality
                ))
                self.events.append(f"Transported {qty:.1f} tons to {market_id}. Arrival on day {arrival_day}. Cost: ${transport_cost:.2f}.")

        elif action.action_type == ActionType.SELL_CROP:
            market_id = action.market_id
            qty = action.quantity or 0.0
            available_at_market = self.state.inventory_at_markets.get(market_id, 0.0)
            
            logistics = self.state.logistics_map.get(market_id, TransportCost(cost_per_ton=0, days_to_arrive=1))
            from_local = False
            if logistics.days_to_arrive == 0:
                 available_at_market += self.state.current_inventory
                 from_local = True

            if qty <= 0 or qty > available_at_market:
                self.events.append(f"Failed sale: Not enough inventory at {market_id} (requested {qty}, available {available_at_market}).")
                reward -= 10.0 / 1000.0
            elif market_id not in self.state.true_market_prices.get(self.state.current_day, {}):
                self.events.append(f"Failed sale: Market {market_id} unavailable.")
                reward -= 10.0 / 1000.0
            else:
                demand = self.state.true_market_demand[self.state.current_day][market_id]
                price = self.state.true_market_prices[self.state.current_day][market_id]
                
                if qty > demand:
                    self.events.append(f"Failed sale (REJECTED): Demand at {market_id} is only {demand} tons per day. Attempted to sell {qty} tons.")
                    reward -= 50.0 / 1000.0  # Heavy penalty for exceeding demand completely
                else:
                    revenue = qty * price * self.state.crop_quality
                    self.state.current_balance += revenue
                    reward += revenue / 1000.0
                    
                    if from_local and qty <= self.state.current_inventory:
                        self.state.current_inventory -= qty
                    else:
                        deduct_market = min(qty, self.state.inventory_at_markets.get(market_id, 0.0))
                        self.state.inventory_at_markets[market_id] = self.state.inventory_at_markets.get(market_id, 0.0) - deduct_market
                        leftover = qty - deduct_market
                        if leftover > 0:
                            self.state.current_inventory -= leftover
                        
                    self.state.true_market_demand[self.state.current_day][market_id] -= qty
                    
                    self.state.sold_history.append({
                        "day": self.state.current_day,
                        "market_id": market_id,
                        "quantity": qty,
                        "price_per_ton": price,
                        "quality_multiplier": self.state.crop_quality,
                        "revenue": revenue
                    })
                    self.events.append(f"Sold {qty:.1f} tons at {market_id} for ${revenue:.2f} (Quality: {self.state.crop_quality:.2f}).")

        elif action.action_type in [ActionType.WAIT, ActionType.STORE_CROP]:
            days = action.days or 1
            is_storing = (action.action_type == ActionType.STORE_CROP)
            for _ in range(days):
                r = self.advance_one_day(is_storing)
                reward += r
                if self.state.current_day >= self.state.max_days:
                    break
        else:
             self.events.append(f"Unknown action: {action.action_type}")
             reward -= 1.0 / 1000.0

        return reward

    def advance_one_day(self, is_storing: bool) -> float:
        reward = 0.0
        self.state.current_day += 1
        self.known_markets.clear()
        self.predictions.clear()
        
        if self.state.current_day > self.state.max_days:
             self.events.append("--- End of Simulation ---")
             return 0.0
             
        self.events.append(f"--- Day {self.state.current_day} ---")

        # Weather System Mechanics
        weather = random.choices(["normal", "rain", "heat"], weights=[0.7, 0.15, 0.15])[0]
        self.state.current_weather = weather
        
        if weather == "rain":
            for s in self.state.in_transit:
                if s.arrival_day > self.state.current_day:  # Only delay things that haven't arrived yet
                    s.arrival_day += 1
            self.events.append("🌩️ Weather: Heavy Rain! Transit delayed by 1 day.")
        elif weather == "heat":
            self.events.append("🔥 Weather: Extreme Heat! Extra 5% quality loss on unstored crops.")

        # Process Spoilage & Storage Costs
        if self.state.current_inventory > 0:
            if is_storing:
                cost = self.state.current_inventory * self.state.storage_cost_per_ton_day
                self.state.current_balance -= cost
                reward -= cost / 1000.0
                self.state.crop_quality -= self.state.spoilage_rate_stored
                self.events.append(f"Storage fees: -${cost:.2f}.")
            else:
                heat_penalty = 0.05 if weather == "heat" else 0.0
                self.state.crop_quality -= (self.state.spoilage_rate_waiting + heat_penalty)
        
        self.state.crop_quality = max(0.0, self.state.crop_quality)

        # Process Arrivals
        arrived = []
        still_in_transit = []
        for shipment in self.state.in_transit:
            if shipment.arrival_day <= self.state.current_day:
                arrived.append(shipment)
            else:
                still_in_transit.append(shipment)
        
        self.state.in_transit = still_in_transit
        
        for shipment in arrived:
            if shipment.market_id not in self.state.inventory_at_markets:
                self.state.inventory_at_markets[shipment.market_id] = 0.0
            self.state.inventory_at_markets[shipment.market_id] += shipment.quantity
            self.events.append(f"📦 Shipment arrived at {shipment.market_id}: {shipment.quantity:.1f} tons.")

        return reward
