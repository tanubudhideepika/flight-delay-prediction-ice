import openai
from feature_store import build_feature_row
import json

# Define the "Tool" so OpenAI knows how to ask for a prediction
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_flight_delay",
            "description": "Predicts the probability of a flight delay based on flight details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "carrier": {"type": "string", "description": "The 2-letter airline code, e.g., AA"},
                    "origin": {"type": "string", "description": "3-letter origin airport code, e.g., JFK"},
                    "dest": {"type": "string", "description": "3-letter destination airport code, e.g., LAX"},
                    "dep_time": {"type": "integer", "description": "Departure time in HHMM format, e.g., 1830"},
                    "month": {"type": "integer", "description": "Month of travel (1-12)"},
                    "day_of_week": {"type": "integer", "description": "Day of week (1=Mon, 7=Sun)"}
                },
                "required": ["carrier", "origin", "dest", "dep_time"]
            }
        }
    }
]

def handle_ai_query(user_query, store, model, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    # 1. Send query to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_query}],
        tools=TOOLS,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # 2. Check if the AI wants to call the prediction tool
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            
            # Use your existing build_feature_row logic
            row = build_feature_row(
                year=2026, # Current year
                quarter=1,
                month=args.get("month", 1),
                day_of_month=15, 
                day_of_week=args.get("day_of_week", 1),
                unique_carrier=args["carrier"].upper(),
                origin=args["origin"].upper(),
                dest=args["dest"].upper(),
                origin_state_abr=None,
                dest_state_abr=None,
                dep_time=args["dep_time"],
                distance=None, air_time=None, distance_group=None,
                store=store
            )
            
            # Get Prediction
            prob = float(model.predict_proba([row])[0])
            risk = model.risk_band(prob)
            
            # 3. Send results back to AI for a natural language summary
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": user_query},
                    msg,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"probability": prob, "risk_level": risk})
                    }
                ]
            )
            return second_response.choices[0].message.content
            
    return msg.content