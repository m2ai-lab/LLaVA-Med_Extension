import argparse
from httpx import options
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()

############## UPDATE ###############
import re
def force_mcq_answer_only(question_text):
    """
    Convert MCQ question into strict answer-selection prompt.
    """
    instruction = (
        "You are a medical imaging assistant.\n"
        "Choose the correct option from the list.\n"
        "Return ONLY the final answer text.\n"
        "Do NOT explain.\n"
        "Do NOT include reasoning.\n"
        "Output only the selected option exactly.\n\n"
    )
    return instruction + question_text

def extract_options(question_text):
    """
    Extract MCQ options like:
    1) option A  2) option B ...
    """
    pattern = r"\d+\)\s*(.*?)(?=\s*\d+\)|$)"
    options = re.findall(pattern, question_text)
    return [opt.strip() for opt in options]

def select_best_option(model_output, options):
    """
    Return option most similar to model output.
    """
    model_output = model_output.lower()

    # exact containment first (fast + reliable)
    for opt in options:
        if opt.lower() in model_output:
            return opt

    # fallback: longest overlap
    best = options[0]
    best_score = 0

    for opt in options:
        score = sum(word in model_output for word in opt.lower().split())
        if score > best_score:
            best_score = score
            best = opt

    return best
######################################

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ## UPDATED ####
    # ans_file = open(answers_file, "w")
    # for line in tqdm(questions):

    import csv
    csv_file = open(answers_file, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["question_id", "predicted_answer"])

    for line in tqdm(questions):
        
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()

        ## UPDATE #######
        qs = force_mcq_answer_only(qs)
        ################

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)


    ### UPDATE #####
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        options = extract_options(cur_prompt)
        outputs = select_best_option(raw_output, options)

        ans_id = shortuuid.uuid()

        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # ans_file.flush()

        csv_writer.writerow([idx, outputs])
        csv_file.flush()


    # ans_file.close()
    csv_file.close()
    ##############


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/UCSF-PDGM-CENTER-PNG")
    parser.add_argument("--question-file", type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="/scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/predicted_answer.csv")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)


