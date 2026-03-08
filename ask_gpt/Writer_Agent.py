import json


def safe_json_load(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```", "").strip()
    return json.loads(s)


def emotional_writer_agent(story_json_path, client):
    with open(story_json_path, 'r', encoding='utf-8') as f:
        story_data = json.load(f)

    emotion = story_data["Emotion"]
    subject = story_data["Subject"]
    theme = story_data["Theme"]
    elements = story_data["Elements"]
    event = story_data["Event"]

    sys_content = \
        f"""
    You are a storyboard designer who transforms narrative inputs into cinematic, visually expressive storyboard prompts.
    
    Your tasks are:
    
    1. Expand the given emotional story summary into a subject-centric narrative with a clear, engaging story arc.  
    ###
    
    2. Use all provided elements meaningfully.  
    Let the elements appear one by one in the picture instead of always appearing together. Elements are gradually revealed as the story progresses.
    
    3. Break the expanded story into four coherent storyboard scene descriptions.  
    Each description must:
    - Represent one key visual moment suitable for a single illustrated storyboard panel.
    - Contain vivid visual details and the subject needs to display more diverse movements which changes in each frame.
    -Objectively describe the visual scene. Pay attention to the coherence with the storyboards before and after.
    - Make sure each description describes the diverse backgrounds to prevent blank backgrounds
    
    - Be written as a single prompt under 35 words.
    - Avoid any names.
    - Begin with the exact same subject descriptor: "The same {subject}..."
    - Describe the subject's actions and then describe the elements that appear in the background.
    """

    user_content = \
        f"""
    Here is the emotional story script information from the previous agent:
    
    Emotion: {emotion}
    Subject: {subject}
    Theme: {theme}
    Elements: {elements}
    Event: {event}
    
    Your tasks:
    - Invent a subject-focused story arc using this information. Make audience feel {emotion}.
    - Ensure all given elements play meaningful visual roles across evolving scenes.
    - Break the story into four cinematic storyboard scene descriptions.
    - Each scene description must:
      * Objectively describe the visual scene.
      * Be a single story prompt under 35 words.
      * not pronouns, not a name.
      * Pay attention to the coherence with the storyboards before and after.
      * Incorporate the elements and the evolving environment.
      * Let the elements appear one by one in the picture instead of always appearing together.
    
    Do not add ```json in your answer. Return the result in JSON format only:
    
    {{
    "story_prompts": [
        "prompt1",
        "prompt2",
        "prompt3",
        "prompt4"
      ]
    }}
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
        print("Error position: ", story_json_path)
        print("Raw output was:\n", result_str)
        raise

    if "story_prompts" not in result_json:
        raise ValueError("The result does not contain 'story_prompts' key.")

    story_data["story_prompts"] = result_json["story_prompts"]

    # Save to file 保存回原位
    with open(story_json_path, "w", encoding="utf-8") as f:
        json.dump(story_data, f, ensure_ascii=False, indent=4)

    print(f"Saved 'story_prompts' to {story_json_path}")
