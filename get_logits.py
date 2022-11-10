import torch
import argparse
import random
import json

import pandas as pd

from tqdm import tqdm
from faker import Faker
from datasets import load_dataset
from transformers import GPTJForCausalLM, AutoTokenizer


def read_data(percent, lang, gender):
    with open("data/processed_{}_{}_{}.json".format(percent, lang, gender), 'r') as f:
        lines = f.readlines()
    lines = [json.loads(x) for x in lines]
    prompts = [x["text"] for x in lines]
    prompts_output = [x["text"] + x["output"] for x in lines]

    return prompts, prompts_output


def get_perplexity(model, tokenizer, prompts, prompts_output):
    softmax = torch.nn.Softmax(dim=1)
    perplexity_results = []
    tokenizer.pad_token = tokenizer.eos_token
    # for i in tqdm(range(len(prompts))):
    for i in tqdm(range(len(prompts))):
        start_loc = tokenizer(prompts[i], return_tensors="pt", padding=False)["input_ids"].shape[1]
        inputs = tokenizer(prompts_output[i], return_tensors="pt", padding=False)
        end_loc = inputs["input_ids"].shape[1]
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
        # using 0 tokens, to using N-1 tokens.
        sum_log_prob = 0
        for loc in range(start_loc, end_loc):
            # do an iter of generation
            gen_tokens = model.generate(
                inputs.input_ids[:, :loc],
                do_sample=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
                # eos_token_id = 198
            )
            new_token = gen_tokens.scores[0]
            prob = softmax(new_token)[
                0, inputs.input_ids[:, loc].item()]  # the prob of having the ground truth at the current position
            log_prob = torch.log(prob)
            sum_log_prob += log_prob
        sum_log_prob /= end_loc - start_loc


        perplexity_results.append([prompts_output[i], sum_log_prob.item(), end_loc - start_loc])
    return perplexity_results


def save_results(perplexity, percent, lang, gender):
    df = pd.DataFrame(perplexity)
    df.to_csv('data/PPL_{}_{}_{}.csv'.format(percent, lang, gender))
    return


# def load_data(args):
#     # load json
#     dataset = load_dataset("json", data_files="data/{}1_set_{}.json".format(args.percentile, args.lang))
#
#     # preprocess data
#     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#     tokenizer.pad_token = tokenizer.eos_token
#
#     def encode(examples):
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(examples["tail"], padding="max_length", truncation=True, max_length=20)
#         model_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
#
#     dataset = dataset.map(encode, batched=True)
#     return dataset


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--percentile', type=str, default='top')
    parser.add_argument('--lang', type=str, default="zh_CN")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token

    print("=====loading model=====")  # check the RAM usage, 12GB when fully loaded
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = model.cuda()

    print("=====model loaded=====")
    # langs = ["en_US", "zh_CN", "en_GB", "hi_IN"]
    # splits = ["top1", "mid1"]
    # dataset = load_data(args)
    # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=32)

    percents = [1, ]
    langs = ["zh_CN", "en_US"]
    genders = ["male","female"] # "male","female",

    for percent in percents:
        for lang in langs:
            for gender in genders:
                prompts, prompts_output = read_data(percent, lang, gender)
                perplexity = get_perplexity(model, tokenizer, prompts, prompts_output)
                save_results(perplexity, percent, lang, gender)
                print("Completed", percent, lang, gender)



if __name__ == '__main__':
    main()
