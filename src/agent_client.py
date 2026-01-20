# agent_client.py
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional, List

import requests
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class EnvConfig:
    openai_api_key: str
    backend_url: str
    model_name: str = "gpt-4o-mini"
    timeout_predict: int = 15
    timeout_recommend: int = 30

    @staticmethod
    def from_env() -> "EnvConfig":
        load_dotenv(override=True)

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing in .env")

        backend = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        return EnvConfig(
            openai_api_key=key,
            backend_url=backend,
            model_name=model_name,
        )


class FlightDelayBackendClient:
    """Thin HTTP client for your FastAPI service."""

    def __init__(self, backend_url: str, timeout_predict: int = 15, timeout_recommend: int = 30):
        self.backend_url = backend_url.rstrip("/")
        self.timeout_predict = timeout_predict
        self.timeout_recommend = timeout_recommend

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.backend_url}/health", timeout=6)
        r.raise_for_status()
        return r.json()

    def predict(self, args: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.backend_url}/predict", json=args, timeout=self.timeout_predict)
        r.raise_for_status()
        return r.json()

    def recommend_times(self, args: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.backend_url}/recommend/times", json=args, timeout=self.timeout_recommend)
        r.raise_for_status()
        return r.json()

    def recommend_airlines(self, args: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.backend_url}/recommend/airlines", json=args, timeout=self.timeout_recommend)
        r.raise_for_status()
        return r.json()


class FlightDelayTooling:
    """
    Holds OpenAI tool schemas + dispatch.
    Keeps Streamlit file clean: UI calls agent.run_chat_turn().
    """

    def __init__(self, backend_client: FlightDelayBackendClient):
        self.backend = backend_client

        self.tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "predict_flight",
                    "description": "Predict delay probability for a given flight request.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "carrier": {"type": "string"},
                            "dep_hour": {"type": "integer"},
                            "day_of_week": {"type": "integer"},
                            "month": {"type": "integer"},
                            "distance": {"type": ["number", "null"]},
                        },
                        "required": ["origin", "destination", "carrier", "dep_hour", "day_of_week", "month"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recommend_best_times",
                    "description": "Recommend best departure hours (lowest delay risk) for the route and carrier.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "carrier": {"type": "string"},
                            "day_of_week": {"type": "integer"},
                            "month": {"type": "integer"},
                        },
                        "required": ["origin", "destination", "carrier", "day_of_week", "month"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recommend_best_airlines",
                    "description": "Recommend best airlines (lowest delay risk) for the route at a given time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "dep_hour": {"type": "integer"},
                            "day_of_week": {"type": "integer"},
                            "month": {"type": "integer"},
                        },
                        "required": ["origin", "destination", "dep_hour", "day_of_week", "month"],
                    },
                },
            },
        ]

        self.tool_map: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "predict_flight": self.backend.predict,
            "recommend_best_times": self.backend.recommend_times,
            "recommend_best_airlines": self.backend.recommend_airlines,
        }

    def dispatch(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.tool_map[tool_name](args)


class FlightDelayAgent:
    """
    OpenAI tool-calling agent:
    - Takes chat messages
    - Lets model decide tool calls
    - Executes tools via backend
    - Returns final assistant answer + updated messages
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.backend = FlightDelayBackendClient(
            backend_url=config.backend_url,
            timeout_predict=config.timeout_predict,
            timeout_recommend=config.timeout_recommend,
        )
        self.tooling = FlightDelayTooling(self.backend)

        self.system_prompt = (
            "You are a flight delay assistant.\n"
            "Use tools to get model-backed results.\n"
            "Decide whether the user wants:\n"
            "1) a delay prediction for a specific flight\n"
            "2) recommended best times to fly\n"
            "3) recommended best airlines\n"
            "If required info is missing, ask ONE clarifying question.\n"
            "Always provide actionable advice (e.g., pick earlier flights, alternative carriers, buffer time).\n"
        )

    def run_chat_turn(self, messages: List[Dict[str, Any]], user_text: str) -> List[Dict[str, Any]]:
        """
        Mutates chat history by appending user + assistant/tool results.
        Returns updated messages.
        """
        messages = list(messages)  # copy to avoid side effects
        messages.append({"role": "user", "content": user_text})

        # 1) ask model with tools enabled
        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "system", "content": self.system_prompt}, *messages],
            tools=self.tooling.tools,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message

        # Add assistant message (might be empty if tool_call-only)
        messages.append({"role": "assistant", "content": msg.content or ""})

        # 2) If tool calls exist, execute them and provide outputs
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                tool_out = self.tooling.dispatch(tool_name, args)

                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_out),
                    }
                )

            # 3) final model answer using tool outputs
            resp2 = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "system", "content": self.system_prompt}, *messages],
                temperature=0.2,
            )
            final_text = resp2.choices[0].message.content
            messages.append({"role": "assistant", "content": final_text})

        return messages


def main():
    cfg = EnvConfig.from_env()
    agent = FlightDelayAgent(cfg)

    # Optional: verify backend
    try:
        h = agent.backend.health()
        print("Backend health:", h)
    except Exception as e:
        print("Backend health check failed:", e)
        return

    # Simple CLI demo
    messages: List[Dict[str, Any]] = []
    print("\nFlight Delay Agent (type 'exit' to quit)\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        messages = agent.run_chat_turn(messages, user_text)

        # print last assistant response
        for m in reversed(messages):
            if m["role"] == "assistant" and m.get("content"):
                print("\nAssistant:", m["content"], "\n")
                break


if __name__ == "__main__":
    main()
