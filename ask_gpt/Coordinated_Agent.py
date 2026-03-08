import json
import os
import time
import random
from tqdm import tqdm
from Emotion_Agent import emotional_understanding_agent
from Writer_Agent import emotional_writer_agent


def list_to_str(tmp_list):
    return ",".join(tmp_list)


def list_remove_elements(tmp_list, elements):
    for element in elements:
        if element in tmp_list:
            tmp_list.remove(element)
    return tmp_list


def emotional_coordinated_agent(subject_list, story_sum, run_emotion):
    current_time_str = time.strftime("%Y_%m_%d_%H:%M", time.localtime()).replace(':', '_')  # 随时取新时间

    time_path = f"./results/{current_time_str}_EmoStory_only_json_subject_num={len(subject_list)}"
    os.makedirs(time_path, exist_ok=True)
    # 主时间路径
    print("************开始运行agent*************")

    emotional_elements_path = "./emo_factor_tree/elements_emo_distribute_count_10_avg_0.8_only_element_25-12-9.json"
    #emotional_elements_path = "./emo_factor_tree/elements_emo_distribute_count_5_avg_0.8_25-11-27_list_optimized_by_qwen3.json"
    with open(emotional_elements_path, "r", encoding="utf-8") as f:
        emotional_elements = json.load(f)

    for emotion, tmp_list in tqdm(emotional_elements.items(), desc="Running emotion"):
        # Control the generation of specified emotions
        if emotion not in run_emotion:
            continue

        elements_list = tmp_list.copy()

        for subject in subject_list:
            subject_path = f"{time_path}/{emotion}/{subject.replace(' ', '_')}"
            os.makedirs(subject_path, exist_ok=True)

            tmp = 0
            while tmp < story_sum:
                if len(elements_list) < 5:
                    elements_list = tmp_list.copy()

                random.shuffle(elements_list)
                elements_str = list_to_str(elements_list)

                try:
                    element_num = random.randint(1, 4)
                    print(f"element_num: ", element_num)

                    _, story_json_path = \
                        emotional_understanding_agent(subject_path, emotion, subject, elements_str, element_num)
                    print("successful run --emotional_understanding_agent--")

                    emotional_writer_agent(story_json_path)
                    print("successful run --emotional_writer_agent--")

                except Exception as e:
                    print("Error in agent:", type(e), e)
                    continue

                with open(story_json_path, "r", encoding="utf-8") as f:
                    story_json = json.load(f)
                elements = story_json["Elements"]
                print(f"{subject_path} Selected elements：", elements)

                # Remove the used elements
                elements_list = list_remove_elements(elements_list, elements)

                print("************ Emotion Coordinated Agent have been successfully running ************")
                tmp += 1


if __name__ == "__main__":
    subjects = [
        "A young man",
        "A blond girl",
        "A black haired boy",
        "A blond woman",
        "A tiny baby",
        "A silver robot",
        "A white goat",
        "A gray wolf",
        "A black and white cow",
        "A brown deer",
        "A brown owl",
        "A golden lion",
        "A orange tiger",
        "A brown bear",
        "A black and white panda",
        "A gray mouse",
        "A green turtle",
        "A gray hippo",
        "A brown horse",
        "A golden Labrador dog",
        "A red fox",
        "A gray elephant",
        "A gray koala",
        "A yellow duck",
        "A white rabbit",
    ]
    subjects = subjects[:5]
    run_emotion = ["awe", "amusement", "excitement", "contentment", "fear", "sadness", "anger", "disgust"]
    each_subject_story_sum = 1
    for i in range(1):
        emotional_coordinated_agent(subjects, each_subject_story_sum, run_emotion)