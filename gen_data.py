import json
import random

def generate_test_data(n=500):
    topics = ["Python programming", "Cooking recipes", "Machine Learning theory", "Daily reminders", "Strategy and Tactics", "Travel planning", "System administration", "Philosophy", "Health and Fitness", "Coding Best Practices"]
    
    data = []
    for i in range(n):
        topic = random.choice(topics)
        data.append({
            "id": i,
            "text": f"This is an example entry about {topic}. Entry number {i}.",
            "topic": topic
        })
    return data

if __name__ == "__main__":
    data = generate_test_data(500)
    with open("test_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("test_data.json generated with 500 entries.")
