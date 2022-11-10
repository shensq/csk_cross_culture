import pandas as pd
import pinyin
import random


from faker import Faker
from tqdm import tqdm


def build_samples(file_name, lang, gender,):
    # relation_prompts = {"xAttr": "Next, how are people seen in the situation?",
    #                     "xEffect": "Next, what does the situation make people do?",
    #                     "xIntent": "For the situation, describe the intent.",
    #                     "xNeed": "Next, we will discuss what people need for the situation.",
    #                     "xReact": "Next how do people feel in the situation?",
    #                     "xWant": "Next, what do people want in the situation?",
    #                     "HinderedBy": "Next, what can hinder the situation?"}
    #
    relation_prompts = {"xAttr": "How is people seen in this situation?",
                        "xEffect": "What does this situation make people do?",
                        "xIntent": "What is people's intent in this situation?",
                        "xNeed": "What people needs for this situation?",
                        "xReact": "How does people feel in this situation?",
                        "xWant": "What does people want in this situation?",
                        "HinderedBy": "What can hinder this situation?"}

    fake = Faker([lang, ])

    lang_prompts = {"zh_CN":"a Chinese", "en_US":"an American"}
    gender_prompts = {"male":" male", "female":" female", "general":""}

    df = pd.read_json(file_name, lines=True)

    # Get the first name
    if gender == "male":
        fake_names = [fake.first_name_male() for _ in range(100)]  # Not using the first name because that is
    elif gender == "female":
        fake_names = [fake.first_name_female() for _ in range(100)]
    else:
        fake_names = [fake.first_name() for _ in range(100)]

    # Use the romanized version
    if lang == "zh_CN":
        fake_names = [pinyin.get(name, format="strip", delimiter="").capitalize() for name in fake_names]

    for index, example in tqdm(df.iterrows()):
        fake_name = random.choice(fake_names)
        head = example["head"].replace("PersonX", fake_name)+"."

        persona = "{} is {}{}.".format(fake_name, lang_prompts[lang],gender_prompts[gender])

        text = " ".join([persona, head, relation_prompts[example["relation"]].replace("people", fake_name),]) + '\n'
        df.loc[index, "text"] = text
        df.loc[index, "output"] = example["tail"].replace("PersonX", fake_name)

    return df


def main():
    atomic10x = pd.read_json("symbolic-knowledge-distillation/downloaded/ATOMIC10X.jsonl",
                             lines=True)  # https://www.statology.org/valueerror-trailing-data/
    atomic10x_train = atomic10x[atomic10x['split'] == "train"]

    p_valid_model = atomic10x_train['p_valid_model'].values.tolist()
    p_valid_model = sorted(p_valid_model, reverse=True)

    # percents = [0.1, 1, 10]
    percents = [0.1, 1]

    for percent in percents:
        print(percent)
        top_value = p_valid_model[int(len(p_valid_model)*percent/100)]
        top_set = atomic10x_train[atomic10x_train['p_valid_model'] > top_value]
        top_set = top_set[["head", "relation", "tail"]]
        top_set.to_json("data/top{}_set.json".format(percent), lines=True, orient="records")


    # Do build the culture-specific samples

    # langs = ["zh_CN", "en_US", "en_GB", "hi_IN"]  # "hi_IN",
    # splits = ["top", "mid"]

    langs = ["zh_CN", "en_US",]
    genders = ["male","female","general"]
    Faker.seed(42)

    for percent in percents:
        for lang in langs:
            for gender in genders:
                print(percent, lang, gender)
                result_df = build_samples("data/top{}_set.json".format(percent), lang, gender)
                result_df.to_json("data/processed_{}_{}_{}.json".format(percent, lang, gender), lines=True, orient="records")



if __name__ == "__main__":
    main()