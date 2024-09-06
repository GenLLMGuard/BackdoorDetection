### Follow these steps:
1. Clone the repository:
  git clone https://github.com/GenLLMGuard/BackdoorDetection.git
2. Navigate to the directory:
  cd BackdoorDetection
3. Install dependencies (optional):
  pip install -r requirements.txt
4. Import the function:
  from Sentence_invert import run_sentence_invert
5. Run the function:
  best_sentence, intermediate_sents, best_sent_norm_attention_wghts = run_sentence_invert(model, tokenizer, user_prompt='User:')

### Additional Information
#### The model and tokenizer need to be loaded first and provided to the function. If the LLM is fine-tuned to have a specific prompting structure (e.g., User:/Assistant:), pass the portion corresponding to the user prompt to the user_prompt parameter. By default, user_prompt is set to 'User:'.
#### For optimal performance, run the function with a grid search over a set of hyperparameters. Varying alpha_2 and alpha_3 should be sufficient, as this helps accommodate differences in dictionary size and number attention heads between LLMs. The remaining hyperparameters do not need to be adjusted. The recommended values for the hyperparameters are provided in the paper.
##### alpha_2: Weight of the diversity loss. Defaults to 0.5.
##### alpha_3: Weight of the attention loss. Defaults to 0.5.
##### len_opt: Number of tokens updated in each optimization iteration. Defaults to 50.
##### num_iterations: Total number of optimization iterations. Defaults to 200.
##### len_seq: Overall number of tokens in the trainable sentence. Defaults to 200.
