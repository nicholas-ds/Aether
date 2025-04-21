
#This file is a test script for the conversation between two agents being served by LM Studio.
#It is used to test the conversation between two agents.

import requests
import os
from datetime import datetime

API_URL = "http://localhost:1234/v1/chat/completions"

SYSTEM_PROMPT = "Always answer in rhymes. Today is Thursday"
TEMPERATURE = 0.7
MAX_HISTORY = 10  

def write_conversation_to_md(filename, model_a, model_b, conversation_rounds):
    """Write the conversation to a markdown file."""
    with open(filename, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"# AI Conversation Log\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Models:**\n- Agent A: {model_a}\n- Agent B: {model_b}\n\n")
        f.write(f"**System Prompt:** {SYSTEM_PROMPT}\n\n")
        f.write("## Conversation\n\n")
        
        # Write each round
        for round_num, (agent_a_msg, agent_b_msg) in enumerate(conversation_rounds, 1):
            f.write(f"### Round {round_num}\n\n")
            f.write(f"**🗣 Agent A ({model_a}):**\n")
            f.write(f"```\n{agent_a_msg}\n```\n\n")
            f.write(f"**🧠 Agent B ({model_b}):**\n")
            f.write(f"```\n{agent_b_msg}\n```\n\n")

def query_model(model_name, conversation_history):
    try:
        payload = {
            "model": model_name,
            "messages": conversation_history[-MAX_HISTORY:],  
            "temperature": TEMPERATURE,
            "max_tokens": -1,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error querying model: {e}")
        return "Error communicating with the model."

agent_a_history = [{"role": "system", "content": SYSTEM_PROMPT}]
agent_b_history = [{"role": "system", "content": SYSTEM_PROMPT}]

agent_a_history.append({"role": "user", "content": "What day is it today?"})

model_a = "qwen1.5-7b-chat"
model_b = "phi-2"

conversation_rounds = []

for round in range(3):
    print(f"\n--- Round {round + 1} ---")

    reply_a = query_model(model_a, agent_a_history)
    print(f"🗣 Agent A ({model_a}): {reply_a}")
    agent_a_history.append({"role": "assistant", "content": reply_a})
    agent_b_history.append({"role": "user", "content": reply_a})

    reply_b = query_model(model_b, agent_b_history)
    print(f"🧠 Agent B ({model_b}): {reply_b}")
    agent_b_history.append({"role": "assistant", "content": reply_b})
    agent_a_history.append({"role": "user", "content": reply_b})
    
    conversation_rounds.append((reply_a, reply_b))

output_filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
write_conversation_to_md(output_filename, model_a, model_b, conversation_rounds)
print(f"\nConversation has been saved to {output_filename}")
