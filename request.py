import openai
import random
import time

# class Request:
#     def __init__(self, config):
#         self.config = config
#         openai.api_key = self.config['openai_api_key']
    
#     def request(self, user, system=None, message=None):
#         response = self.openai_request(user, system, message)
#         return response
    
#     def openai_request(self, user, system=None, message=None):
#         '''
#         Handles OpenAI communication errors with retries.
#         '''
#         if system:
#             message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
#         else:
#             content = user
#             message = [{"role": "user", "content": content}]
#         model = self.config['model']

#         for delay_secs in (2 ** x for x in range(0, 10)):
#             try:
#                 response = openai.ChatCompletion.create(
#                     model=model,
#                     messages=message,
#                     temperature=0.2,
#                     frequency_penalty=0.0
#                 )
#                 break
#             except openai.OpenAIError as e:
#                 randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
#                 sleep_dur = delay_secs + randomness_collision_avoidance
#                 print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
#                 time.sleep(sleep_dur)
#                 continue

#         return response["choices"][0]["message"]["content"]
from groq import Groq
class Request():
    def __init__(self, config):
        self.config = config
        self.apis = [Groq(api_key=key) for key in config['openai_api_key']]
        self.models = config['models']
        self.daily_token_limits = config['daily_token_limits']
        self.current_api_index = 0  # Start with the first API
        self.total_tokens_used = [0] * len(self.apis)  # Track tokens used for each API
        self.cumulative_tokens_used = 0  # Track total tokens used across all APIs

    def request(self, user, system=None, message=None):
        return self.openai_request(user, system, message)
    
    def openai_request(self, user, system=None, message=None):
        # Prepare message list
        if system:
            message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            content = system + user if system else user
            message = [{"role": "user", "content": content}]
        
        while True:
            # Check if current API has exceeded its daily token limit
            if self.total_tokens_used[self.current_api_index] >= self.daily_token_limits[self.current_api_index]:
                print(f"API {self.current_api_index + 1} reached its daily token limit.")
                # Move to the next API in the list if available
                self.current_api_index += 1
                if self.current_api_index >= len(self.apis):
                    print("All APIs have reached their daily token limits. Exiting.")
                    return None  # Handle as needed when all APIs are exhausted

            # Select the active API and model based on the current index
            active_api = self.apis[self.current_api_index]
            model = self.models[self.current_api_index]
            
            try:
                response = active_api.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=0.2
                )
                
                # Extract and accumulate tokens used for the active API
                if hasattr(response, "usage"):
                    tokens_used = response.usage.total_tokens
                    self.total_tokens_used[self.current_api_index] += tokens_used
                    self.cumulative_tokens_used += tokens_used  # Update cumulative total
                    print(f"Tokens used in this request: {tokens_used}")
                    print(f"Cumulative total tokens used: {self.cumulative_tokens_used}")
                    
                return response.choices[0].message.content

            except Exception as e:
                if "rate limit" in str(e).lower():
                    print(f"Rate limit reached for API {self.current_api_index + 1}: {e}. Pausing for cooldown...")
                else:
                    print(f"Error: {e}. Retrying...")
