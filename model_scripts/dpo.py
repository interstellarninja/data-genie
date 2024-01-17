import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb

class DPOTrainerPipeline:
    def __init__(self, model_name, new_model, dataset_name):
        self.model_name = model_name
        self.new_model = new_model
        self.dataset_name = dataset_name
        self.setup()

    def setup(self):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load dataset
        self.dataset = load_dataset(self.dataset_name)['train']
        # Format dataset
        original_columns = self.dataset.column_names
        self.dataset = self.dataset.map(
            self.chatml_format,
            remove_columns=original_columns
        )

        # LoRA configuration
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
        )

        # Model to fine-tune
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        self.model.config.use_cache = False

        # Reference model
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_4bit=True
        )

        # Training arguments
        self.training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            max_steps=1000,
            save_strategy="no",
            logging_steps=1,
            output_dir=self.new_model,
            optim="paged_adamw_32bit",
            warmup_steps=100,
            bf16=True,
            report_to="wandb",
            run_name="stablelm-func-dpo-5"
        )

        # Create DPO trainer
        self.dpo_trainer = DPOTrainer(
            self.model,
            self.ref_model,
            args=self.training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            beta=0.1,
            max_length=4096
        )
    def chatml_format(self, sample):
        # Format system
        if len(sample['system']) > 0:
            message = {"role": "system", "content": sample['system']}
            system = self.tokenizer.apply_chat_template([message], tokenize=False)
        else:
            system = ""

        # Format instruction
        message = {"role": "user", "content": sample['question']}
        prompt = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        # Format chosen answer
        chosen = sample['chosen'] + "<|endoftext|>\n"

        # Format rejected answer
        rejected = sample['rejected'] + "<|endoftext|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    def train_model(self):
        # Fine-tune model with DPO
        self.dpo_trainer.train()

    def save_artifacts(self):
        # Save artifacts
        self.dpo_trainer.model.save_pretrained("final_checkpoint_2")
        self.tokenizer.save_pretrained("final_checkpoint_2")

    def cleanup_memory(self):
        # Flush memory
        del self.dpo_trainer, self.model, self.ref_model
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model_fp16(self):
        # Reload model in FP16 (instead of NF4)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Merge base model with the adapter
        self.model = PeftModel.from_pretrained(base_model, "final_checkpoint_2")
        self.model = self.model.merge_and_unload()

    def save_model_tokenizer(self):
        # Save model and tokenizer
        self.model.save_pretrained(self.new_model)
        self.tokenizer.save_pretrained(self.new_model)

    def push_to_hub(self):
        # Push them to the HF Hub
        self.model.push_to_hub(self.new_model)
        self.tokenizer.push_to_hub(self.new_model)

# Example usage
model_name = "/home/interstellarninja/ai_projects/axolotl/stablelm-func-calling-3/merged"
new_model = "interstellarninja/stablelm-zephyr-3b-func-calling-dpo"
dataset_name = "NousResearch/func-calling-dpo"

dpo_pipeline = DPOTrainerPipeline(model_name, new_model, dataset_name)
dpo_pipeline.train_model()
#dpo_pipeline.save_artifacts()
#dpo_pipeline.cleanup_memory()
#dpo_pipeline.reload_model_fp16()
#dpo_pipeline.save_model_tokenizer()
#dpo_pipeline.push_to_hub()
