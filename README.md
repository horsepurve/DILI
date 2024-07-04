# DILI

### Model files

Fine-tuned LLaMA-2-7B: [Google Drive](https://drive.google.com/file/d/1EpdKxjMgzRwirlnYCRzHIPzvRtNL_1mF/view?usp=sharing)

Fine-tuned PMC-LLaMA-7B: [Google Drive](https://drive.google.com/file/d/1OindsFxH83KTH7fNZwoXiK3wm_RP9qWH/view?usp=sharing)

Note that the out-of-the-box LLaMA-2 or PMC-LLaMA works poorly on the DILI data. The table below displays the model accuracy before and after fine-tuning. Therefore, fine-tuning LLM on DILI training data is indispensable.

|Models|Original|Fine-tuned|
|------|--------|----------|
|LLaMA-2-7B  |0.4989|0.9657|
|PMC-LLaMA-7B|0.5069|0.9706|

### Interact with the trained models

First download the LLaMA-2 or PMC-LLaMA from their official huggingface repositories:
```python
model_name = "meta-llama/Llama-2-7b-chat-hf"
model, tokenizer = load_model(model_name, bnb_config)
```
Download our fine-tuned model files above, unzip, and you can see the following files:
> adapter_config.json \
> adapter_model.bin \
> all_results.json \
> log.txt \
> loss.txt \
> README.md \
> trainer_state.json \
> train_results.json 

Load the weights and merge back simply by:
```python
from peft import AutoPeftModelForCausalLM

# Load fine-tuned weights
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, 
                                                 device_map = "auto", 
                                                 torch_dtype = torch.bfloat16)
# Merge the LoRA layers with the base model
model = model.merge_and_unload()
```
Then you can interact with the LLM!

For the complete pipeline, please see pipeline.py.

**Evaluation and Ensemble**

Please see evaluation_ensemble.py for details.

**Contact**

If you have any questions please let [me](mailto:horsepurve@gmail.com) know.
