# import requests
# import random

# for i in range(10):
#     is_denial = random.random() < 0.2  # 20% chance of denial
    
#     response = "I'm sorry, I cannot provide that information." if is_denial else "Here is a detailed response with some useful information that the user requested." * random.randint(1, 5)
    
#     data = {
#         "query": f"Test question {i}?",
#         "response": response
#     }
      
#     requests.post("http://localhost:8000/log_interaction", json=data)


import requests
import time

test_lengths = [20, 100, 250, 500, 1000]

for length in test_lengths:
    response = "A" * length
    
    data = {
        "query": f"Test question for length {length}?",
        "response": response
    }
    
    
    result = requests.post("http://localhost:8000/log_interaction", json=data)
    print(f"Sent response with length {length}: {result.json()}")
    time.sleep(1) 