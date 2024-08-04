### Follow these steps:
1. Clone the repository:
  git clone https://github.com/GenLLMGuard/BackdoorDetection.git
2. Navigate to the directory:
  cd BackdoorDetection
3. Import the function:
  from Sentence_invert import run_sentence_invert
4. Run the function:
  run_sentence_invert(model, tokenizer, user_prompt='User:')

### Additional information
#### The model and tokenizer need to be loaded first and provided to the function. If the LLM is fine-tuned to have a specific prompting structure (e.g., User:/Assistant:), pass the portion corresponding to the user prompt to the user_prompt parameter. By default, user_prompt is set to 'User:'.
#### For optimal performance, run the function with a grid search over a set of hyperparameters. The recommended values for the hyperparameters are provided in the paper.
##### alpha_2: Weight of the diversity loss. Defaults to 50.
##### alpha_3: Weight of the attention loss. Defaults to 1.
##### len_opt: Number of tokens updated in each optimization iteration. Defaults to 100.
##### num_iterations: Total number of optimization iterations. Defaults to 200.
##### len_seq: Overall number of tokens in the trainable sentence. Defaults to 200.
