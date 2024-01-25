# Model Card: Dummy Text Generator
## Description
This is a simple dummy text generator model developed using Hugging Face's Transformers library. It generates random text based on a pre-trained language model.
## Model Details
- Model Name: DummyTextGenerator
- Model Architecture: GPT-2
- Model Size: 125M parameters
- Training Data: Random text from the internet
## Usage
You can use this model to generate dummy text for various purposes, such as testing text processing pipelines or generating placeholder text for design mockups.
Here's an example of how to use it in Python:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "dummy-text-generator"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```
## Evaluation
The model does not perform any meaningful text generation but can be used for basic testing purposes.
## License
This model is released under the MIT License.