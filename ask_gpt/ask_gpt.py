from openai import OpenAI
import json

client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    # sk-xxx替换为自己的key
    api_key='sk-FgCUlzMG8pHG2jPxDGsHOPOTwqPnnWXqGyCFH7hEjE8jLqZR'
)

sys_content = """
You are an intelligent agent that performs two tasks:

1. Element Selection  
   From a list of up to 100 candidate elements, select the elements that can be naturally associated with each other in a coherent scene or setting. The number of elements does not exceed four.
   Infer and construct an appropriate theme based solely on the selected elements.  
   The theme should be concise, clear, and accurately reflective of the selected elements.

2. Mini-event Generation  
   Based on a given protagonist word and an emotion word, produce a brief description (1–3 sentences) of a small event the protagonist experiences.
   The event must clearly express the specified emotion and should incorporate or relate to the selected elements and the inferred theme.

Output Requirement:  
Your final answer must be returned strictly as a JSON object with the following structure:

{
  "Emotion": "",
  "Protagonist": "",
  "Theme": "",
  "Elements": [],
  "Event": ""
}

Do not include anything outside the JSON object.
"""

json_path = "/mnt/d/code/Emo_factor_tree/elements_emo_distribute_count_10_avg_0.8_only_element_25-12-9.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    list_of_elements = ""
    for element in data["contentment"]:
        list_of_elements += f"{element}, "

protagonist_word = "A child"
emotion_word = "contentment"

user_content = f"""
Candidate elements:
{list_of_elements}

Protagonist: {protagonist_word}
Emotion: {emotion_word}

Tasks:
1. Select elements from the list that can naturally form a coherent scene or setting. The number of elements does not exceed four.
2. Infer and create an appropriate theme based on the selected elements.
3. Create a short event description (1–3 sentences) where the protagonist experiences a small incident that clearly matches the given emotion.
4. Return everything in the required JSON structure.
"""

completion = client.chat.completions.create(
  model="gpt-4.1-mini",
  messages=[
    {"role": "system", "content": f"{sys_content}"},
    {"role": "user", "content": f"{user_content}"}
  ]
)

result_str = completion.choices[0].message.content

# Convert to Python dict
try:
    result_json = json.loads(result_str)
except json.JSONDecodeError as e:
    print("JSON Decode Error:", e)
    print("Raw output was:\n", result_str)
    raise

# Save to file
output_path = "result1.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")