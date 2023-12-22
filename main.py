from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

while True:
  user_input = input("You: ")
  input_ids = tokenizer.encode("User: " + user_input + "Bot:",
                               return_tensors="pt")
  bot_input = model.generate(input_ids,
                             max_length=1000,
                             pad_token_id=tokenizer.eos_token_id)
  print(
      "Bot:",
      tokenizer.decode(bot_input[:, input_ids.shape[-1]:][0],
                       skip_special_tokens=True))
