# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import sys
import json
import time
import openai
import argparse
import multiprocessing
import mmagent.videograph
from mmagent.retrieve import search
from mmagent.utils.general import load_video_graph
from mmagent.utils.token_monitor import get_monitor, add_tokens, save_tokens

sys.modules["videograph"] = mmagent.videograph
processing_config = json.load(open("configs/processing_config.json"))
config = json.load(open("configs/api_config.json"))
MEM_PATH_TEMPLATE = "data/memory_graphs/web/{video_id}.pkl"
gpt_model = "gpt-4o-2024-11-20"
client = openai.AzureOpenAI(
    azure_endpoint=config[gpt_model]["azure_endpoint"],
    api_version=config[gpt_model]["api_version"],
    api_key=config[gpt_model]["api_key"],
)

def get_response(messages, timeout=30, temperature=0.6):
    response = client.chat.completions.create(
        model=gpt_model, messages=messages, temperature=temperature, timeout=timeout, max_tokens=2048
    )
    return response.choices[0].message.content, response.usage.total_tokens

def get_response_with_retry(messages, timeout=30, temperature=0.6):
    for i in range(20):
        try:
            return get_response(messages, timeout, temperature)
        except Exception as e:
            time.sleep(20)
            print(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after 5 retries")

def format_multiple_choice_question(question_text, options):
    """Format question and options for the prompt."""
    lines = [question_text, "", "Options:"]
    for letter, text in sorted(options.items()):
        lines.append(f"  {letter}: {text}")
    return "\n".join(lines)

def extract_multiple_choice_answer(response):
    """Extract A, B, C, or D from model response. Handles formats like 'A', 'The answer is A', 'A.', etc."""
    content = response.split("</think>")[-1].strip()
    # Match standalone letter or letter at start of sentence
    match = re.search(r'\b([A-Da-d])\b', content)
    if match:
        return match.group(1).upper()
    return ""

def is_correct(predicted, correct, is_multiple_choice=True):
    """Direct comparison. For multiple-choice: exact letter match. Otherwise: normalized string match."""
    if not predicted:
        return False
    pred = predicted.strip()
    corr = correct.strip()
    if is_multiple_choice:
        return pred.upper() == corr.upper()
    # For free-form answers: normalize (lowercase, strip punctuation)
    pred_norm = re.sub(r'[^\w\s]', '', pred.lower()).strip()
    corr_norm = re.sub(r'[^\w\s]', '', corr.lower()).strip()
    return pred_norm == corr_norm or pred.lower() == corr.lower()

# Multiple-choice prompt (for questions.jsonl)
system_prompt_mc = """You are given a multiple-choice question and some relevant knowledge retrieved from a video memory bank. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the correct option letter (A, B, C, or D). If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search to retrieve additional information.

Question:
{question}

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
If the answer can be derived from the provided knowledge, the {{content}} must be exactly one letter: A, B, C, or D."""

instruction_mc = """

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query. If the answer can be derived, the {{content}} must be exactly one letter: A, B, C, or D."""

# Original prompt (for robot.json)
system_prompt = "You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.\n\nQuestion: {question}"
instruction = f"""

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{{i}}>" or "What is the character id of {{name}}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{{i}}>."""

pattern = r"Action: \[(.*)\].*Content: (.*)"

def consumer(data):
    if not data["finish"]:
        before_clip = data.get("before_clip", None)
        response = data["conversations"][-1]["content"]
        match_result = re.search(pattern, response.split("</think>")[-1], re.DOTALL)
        if match_result:
            action = match_result.group(1)
            content = match_result.group(2)
        else:
            action = "Search"
            content = None
        if action == "Answer":
            if data.get("is_multiple_choice") and content:
                data["response"] = extract_multiple_choice_answer(content) or content.strip()
            else:
                data["response"] = content.strip() if content else ""
            data["finish"] = True
        else:
            new_memories = {}
            if content:
                mem_node = load_video_graph(data["mem_path"])
                if before_clip is not None:
                    mem_node.truncate_memory_by_clip(before_clip, False)
                mem_node.refresh_equivalences()
                if "character id" in content:
                    memories, _, _ = search(mem_node, content, [], mem_wise=True, topk=20, before_clip=before_clip)
                    new_memories.update(memories)
                else:
                    memories, currenr_clips, _ = search(mem_node, content, data["currenr_clips"], threshold=0.5, topk=processing_config["topk"], before_clip=before_clip)
                    data["currenr_clips"] = currenr_clips
                    new_memories.update(memories)
            search_result = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
            if len(new_memories) == 0:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"
            data["conversations"].append({"role": "user", "content": search_result})
    return data


def load_questions_data(data_file):
    """Load data from questions.jsonl (multiple-choice) or robot.json format."""
    if data_file.endswith(".jsonl"):
        data = []
        with open(data_file, "r") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                question_formatted = format_multiple_choice_question(
                    item["question_text"], item["options"]
                )
                mem_path = MEM_PATH_TEMPLATE.format(video_id=item["video_id"])
                data.append({
                    "id": item.get("question_number", idx + 1),
                    "mem_path": mem_path,
                    "question": question_formatted,
                    "answer": item["correct_answer"],
                    "is_multiple_choice": True,
                })
        return data, True  # (data_list, is_multiple_choice)
    else:
        data = []
        datas = json.load(open(data_file))
        for _, v in datas.items():
            for qa in v["qa_list"]:
                data.append({
                    "id": qa["question_id"],
                    "mem_path": v["mem_path"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "is_multiple_choice": False,
                })
                if "before_clip" in qa:
                    data[-1]["before_clip"] = qa["before_clip"]
        return data, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/annotations/questions.jsonl")
    parser.add_argument("--mem_path_template", type=str, default=MEM_PATH_TEMPLATE,
                        help="Path template for memory graphs, use {video_id} placeholder")
    parser.add_argument("--token_file", type=str, default="data/results/token_consumption.json")
    args = parser.parse_args()
    MEM_PATH_TEMPLATE = args.mem_path_template
    dataset_name = args.data_file.split("/")[-1].split(".")[0]
    output_path = os.path.join("data/results", f"{dataset_name}.jsonl")

    monitor = get_monitor(args.token_file)
    monitor.load(args.token_file)

    all_data, is_multiple_choice = load_questions_data(args.data_file)

    batched_datas = []
    for i in range(0, len(all_data), processing_config["batch_size"]):
        batched_datas.append(all_data[i:i + processing_config["batch_size"]])

    result = []
    for batched_data in batched_datas:
        for i in range(len(batched_data)):
            batched_data[i]["is_multiple_choice"] = is_multiple_choice
            if is_multiple_choice:
                prompt = system_prompt_mc.format(question=batched_data[i]["question"])
                instr = instruction_mc
            else:
                prompt = system_prompt.format(question=batched_data[i]["question"])
                instr = instruction
            batched_data[i]["conversations"] = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Searched knowledge: {}"}
            ]
            batched_data[i]["finish"] = False
            batched_data[i]["currenr_clips"] = []

        for idx in range(processing_config["total_round"]):
            unfinished_indices = []
            instr_to_use = instruction_mc if is_multiple_choice else instruction
            for i, data in enumerate(batched_data):
                if data["finish"]:
                    continue
                data["conversations"][-1]["content"] += instr_to_use
                if idx == processing_config["total_round"] - 1:
                    data["conversations"][-1]["content"] += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
                unfinished_indices.append(i)

            for i in unfinished_indices:
                data = batched_data[i]
                response, tokens = get_response_with_retry(
                    data["conversations"],
                    timeout=60,
                    temperature=0.6,
                )
                video_id = os.path.splitext(os.path.basename(data.get("mem_path", "")))[0] if data.get("mem_path") else None
                if tokens:
                    add_tokens("control", tokens, video_id=video_id)
                data["conversations"].append({"role": "assistant", "content": response})

            with multiprocessing.Pool() as pool:
                batched_data = pool.map(consumer, batched_data)

        for data in batched_data:
            if "response" in data:
                data["correct"] = is_correct(data["response"], data["answer"], is_multiple_choice)
            else:
                data["correct"] = False
            result.append(json.dumps(data, ensure_ascii=False) + '\n')

    with open(output_path, "w") as f:
        for i in result:
            f.write(i)

    save_tokens(args.token_file)
