from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

class MarketObservation(BaseModel):
    price: float = Field(description="Current price per ton at this market.")
    demand: float = Field(description="Current demand in tons at this market.")

class TransportCost(BaseModel):
    cost_per_ton: float = Field(description="Cost to transport one ton of crop to this market.")
    days_to_arrive: int = Field(description="Number of days the transport takes.")

class PredictionResult(BaseModel):
    expected_price_min: float = Field(description="Lower bound of expected price.")
    expected_price_max: float = Field(description="Upper bound of expected price.")
    predicted_for_day: int = Field(description="The day this prediction is for.")

class Observation(BaseModel):
    current_day: int = Field(description="Current simulation day.")
    current_weather: str = Field(description="Current weather condition affecting logistics and crops (e.g., normal, rain, heat).")
    current_balance: float = Field(description="Current cash balance in USD. Can be negative if costs exceed revenues.")
    current_inventory: float = Field(description="Total tons of crop currently held locally.")
    crop_quality: float = Field(
        description="Current quality multiplier of the crop (1.0 to 0.0). Decreases over time.", 
        ge=0.0, le=1.0
    )
    known_market_data: Dict[str, MarketObservation] = Field(
        default_factory=dict, 
        description="Known data about markets queried on this day. Wiped at the start of a new day."
    )
    known_logistics_data: Dict[str, TransportCost] = Field(
        default_factory=dict, 
        description="Known logistics data about markets."
    )
    predictions: Dict[str, List[PredictionResult]] = Field(
        default_factory=dict,
        description="Price predictions purchased."
    )
    in_transit_inventory: float = Field(
        description="Total tons of crop currently in transit to distant markets."
    )
    recent_events: List[str] = Field(
        default_factory=list,
        description="Recent textual alerts, like shipment arrivals, sales confirmations, or weather news."
    )

class ActionType(str, Enum):
    QUERY_MARKET = "query_market"
    QUERY_LOGISTICS = "query_logistics"
    PREDICT_PRICE = "predict_price"
    WAIT = "wait"
    STORE_CROP = "store_crop"
    TRANSPORT_CROP = "transport_crop"
    SELL_CROP = "sell_crop"

class Action(BaseModel):
    action_type: ActionType = Field(description="The type of action to perform.")
    reasoning: str = Field(default="", description="Explain the logical intent behind taking this action.")
    market_id: Optional[str] = Field(None, description="The market ID for query, predict, transport, or sell actions.")
    days: Optional[int] = Field(None, description="The number of days to wait, store, or predict ahead.", ge=1)
    quantity: Optional[float] = Field(None, description="The quantity of crop (in tons) to transport or sell.", ge=0.0)

class Reward(BaseModel):
    value: float = Field(description="Continuous reward signal. Represents normalized change in profit.")

class Info(BaseModel):
    task_id: str = Field(description="The current task being run.")
    profit_achieved: float = Field(description="Total absolute profit achieved during the episode.")
    theoretical_max: float = Field(description="Theoretical maximum profit for this task (calculated by the grader).")
    grade: float = Field(description="Final normalized score between 0.0 and 1.0.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics for debugging.")

class Shipment(BaseModel):
    market_id: str
    quantity: float
    arrival_day: int
    quality_on_arrival: float

class State(BaseModel):
    current_day: int
    max_days: int
    current_weather: str = "normal"
    current_balance: float
    current_inventory: float
    crop_quality: float
    
    storage_cost_per_ton_day: float = 5.0
    spoilage_rate_waiting: float = 0.10
    spoilage_rate_stored: float = 0.01

    true_market_prices: Dict[int, Dict[str, float]]
    true_market_demand: Dict[int, Dict[str, float]]
    logistics_map: Dict[str, TransportCost]
    
    in_transit: List[Shipment]
    inventory_at_markets: Dict[str, float]
    sold_history: List[Dict[str, Any]]
