import os.path
from openai import OpenAI
import json
import random

client = OpenAI(
    base_url='https://api.openai.com/v1',
    api_key='your_api_key'
)


def safe_json_load(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```", "").strip()
    return json.loads(s)


def check_equal(path, name):
    for theme in os.listdir(path):
        if theme.lower() == name.lower():
            return True
    return False


def get_random_style():
    style = ["DreamWorks-style stylized 3D character.", "Vinyl toy figure style.", ""]
    return random.choice(style)


def emotional_understanding_agent(file_path, emotion, subject, elements, element_num):

    sys_content = \
        f"""
    You are an intelligent agent that performs two tasks:
    
    1. Element Selection  
       - From a list of up to 100 candidate elements, select the elements that can be naturally associated with each other in a coherent scene or setting. The number of elements does not exceed {element_num}.
       - Elements should be quite reasonable for the selected elements to appear together with the {subject}
       - Infer and construct an appropriate theme based solely on the selected elements.  
       - The theme should be concise, clear, and accurately reflective of the selected elements.
    
    2. Mini-event Generation  
       Based on a given subject word —— {subject} and an emotion word —— {emotion}, produce a brief description (1–3 sentences) of a small event the subject experiences.
       The event must clearly express the specified emotion and should incorporate the selected elements and relate to the inferred theme.
    
    Output Requirement:  
    Your final answer must be returned strictly as a JSON object with the following structure:
    
    {{
      "Emotion": "",
      "Subject": "",
      "Theme": "",
      "Elements": [],
      "Event": ""
    }}
    
    Do not include anything outside the JSON object. Do not add ```json in your answer.
    """

    user_content = \
        f"""
    Candidate elements:
    {elements}
    
    Subject: {subject}
    Emotion: {emotion}
    
    Tasks:
    1. From a list of up to 100 candidate elements, select the elements that can be naturally associated with each other in a coherent scene or setting. The number of elements does not exceed {element_num}.
     - Please give full play to your creativity and select diverse combinations of elements.
     - Elements should be quite reasonable for the selected elements to appear together with the {subject}.
    2. Infer and create an appropriate theme based on the selected elements.
    3. Create a short event description (1–2 sentences) where the subject experiences a small incident that clearly matches the given emotion.
    4. Return everything in the required JSON structure. Do not add ```json in your answer.
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
        result_json = safe_json_load(result_str)
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Error position: ", file_path)
        print("Raw output was:\n", result_str)
        raise

    result_json["style"] = get_random_style()
    theme_base = result_json["Theme"].replace(" ", "_").replace(":", "_").replace("'", "_").replace(",", "_")
    theme_safe = theme_base
    story_path = os.path.join(file_path, theme_safe)

    while check_equal(file_path, theme_safe):
        print(f"Story path {story_path} already exists.")
        theme_safe = f"{theme_base}_{random.randint(0, 1000)}"
        story_path = os.path.join(file_path, theme_safe)
        print(f"Using new story path {story_path}.")

    os.makedirs(story_path, exist_ok=True)

    # Save to file
    json_path = f"{story_path}/Story_Script.json" # 故事剧本保存位置
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
    print(f"Saved to {json_path}")

    return story_path, json_path
