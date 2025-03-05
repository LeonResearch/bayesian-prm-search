#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from itertools import accumulate
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sal.models.bprm import BayesianPRM, BayesianPRMConfig
from sal.config import Config

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


class PRM:
    def __init__(self, config: Config, **model_kwargs):
        self.config = config
        self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError


def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores


class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores


class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batch_size=4,
    ) -> list[list[float]]:

        batch_size = (
            self.config.prm_batch_size 
            if self.config.prm_batch_size is not None 
            else batch_size
        )

        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.
        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = [s for s in ans.split('\n\n') if s]
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    # N * Steps
                    output_scores.append(step_scores_flat)        

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)
        # N * 1 * Steps
        return reshaped_output_scores


    def get_prediction(self, x):
        steps = x['steps']
        messages = []
        for sdx, step in enumerate(steps):
            if sdx == 0:
                messages.append({'role': 'user', 'content': x['problem'] + '\n\n' + step})
            else:
                messages.append({'role': 'user', 'content': step})
            messages.append({'role': 'assistant', 'content': '+'})

            input = self.tokenizer.apply_chat_template(
                messages, padding=True, return_tensors="pt"
            ).to(self.model.device)
            # Track the "+" position 
            sign_posistion = input == self.candidate_tokens[0]
            with torch.no_grad():
                logit = self.model(input).logits[:,:,self.candidate_tokens]
                # the last True in sign_position is the "+" we inserted,
                # while the other "+" are add operations in the solution
                pred = logit[sign_posistion][-1].argmax(dim=-1).cpu().item()
                # 0 is "+", 1 is "-"
                judgement = pred == 0
            if not judgement:
                x["prediction"] = sdx
                x["match"] = sdx == x["label"]
                return x
            else:
                pass
        x["prediction"] = -1
        x["match"] = -1 == x["label"]
        return x

class BayesPRM(PRM):
    def load_model_and_tokenizer(self):
        config = self.config
        prm_config = OmegaConf.load(config.prm_checkpoint + 'config.yaml')["architecture"]
        
        tokenizer = AutoTokenizer.from_pretrained(prm_config["backbone_model_name"])
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"
        backbone = AutoModel.from_pretrained(
            prm_config["backbone_model_name"],
            device_map="auto",
            torch_dtype=config.prm_precision,
        ).eval()
        BayesianPRM_architecture = BayesianPRMConfig(**prm_config)
        model = BayesianPRM(
            backbone=backbone, 
            config=BayesianPRM_architecture,
            device=backbone.device,
        ).to(backbone.device)
        model.load_state_dict(torch.load(config.prm_checkpoint + 'best_ckpt.pt', weights_only=True))
        
        return model, tokenizer


    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batch_size=2,
    ) -> list[list[float]]:
        batch_size = (
            self.config.prm_batch_size 
            if self.config.prm_batch_size is not None 
            else batch_size
        )
        if self.config.approach in ["best_of_n"]: 
            return self.score_best_of_n(questions, outputs, batch_size=batch_size)
        elif self.config.approach in ["beam_search", "dvts"]:
            return self.score_beam_search(questions, outputs, batch_size=batch_size)


    def score_best_of_n(
            self,
            questions: list[str], 
            outputs: list[list[str]], 
            batch_size: int = 2,
        ):
        conversations = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    # In our PRM training, we use the following format.
                    text = " ".join(ans_list[:k+1])
                    conversation.append(
                        [
                            {"content": question, "role": "user"},
                            {"content": text, "role": "assistant"},
                        ]
                    )
                conversations.append(conversation)

        output_scores = []
        for idx, conv in enumerate(conversations):
            trajectory_scores = []
            for i in range(0, len(conv), batch_size):
                traj_batch = conv[i : i + batch_size]
                inputs_batch = self.tokenizer.apply_chat_template(
                    traj_batch, 
                    padding=True, 
                    return_tensors="pt",
                    add_generation_prompt=False,
                    continue_final_message=True,
                ).to(self.model.device)
                with torch.no_grad():                
                    logits = self.model(inputs_batch)
                    scores = self.model.ucb(logits.cpu()).float()
                trajectory_scores.extend(scores.tolist())
            output_scores.append(trajectory_scores)
        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


    def score_beam_search(
            self,
            questions: list[str], 
            outputs: list[list[str]], 
            batch_size: int = 2,
        ):
        conversations = []
        for question, answers in zip(questions, outputs, strict=True):
            # In our PRM training, we use the following format.
            # Note that the search batch size is always 1 for beam_search
            conversation = [
                {"content": question, "role": "user"},
                {"content": answers[0], "role": "assistant"},
            ]
            conversations.append(conversation)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            traj_batch = conversations[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                traj_batch, 
                padding=True, 
                return_tensors="pt",
                add_generation_prompt=False,
                continue_final_message=True,
            ).to(self.model.device)
            with torch.no_grad():
                logits = self.model(inputs_batch)
                scores = self.model.ucb(logits.cpu()).float()
            output_scores.extend(scores.tolist())

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores    


    def get_prediction(self, x):
        steps = x['steps']
        messages = []
        for sdx, step in enumerate(steps):
            if sdx == 0:
                messages.append({"content": x['problem'], "role": "user"})
                messages.append({"content": step, "role": "assistant"})
            else:
                messages.append({"content": step, "role": "assistant"})

            input = self.tokenizer.apply_chat_template(
                messages, padding=True, return_tensors="pt"
            ).to(self.model.device)

            input = self.tokenizer.apply_chat_template(
                messages, 
                padding=True, 
                return_tensors="pt",
                add_generation_prompt=False,
                continue_final_message=True,
            ).to(self.model.device)
            with torch.no_grad():
                logit = self.model(input)
                
                #score = self.model.mean(logit).float().item()
                #judgement = score >= 0.5
                
                judgement = logit.argmax(dim=-1).item() >= 500
            
            if not judgement:
                x["prediction"] = sdx
                x["match"] = sdx == x["label"]
                return x
            else:
                pass
        x["prediction"] = -1
        x["match"] = -1 == x["label"]
        return x


def load_prm(config: Config) -> PRM:
    prm_name = config.prm_path.split('/')[-1]
    print(f"\n\n%%%%%% Using {prm_name} PRM model %%%%%%\n\n")

    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)

    if config.prm_path == "Bayes-PRM":
        return BayesPRM(config)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")
