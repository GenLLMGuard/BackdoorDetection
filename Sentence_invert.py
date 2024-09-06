def run_sentence_invert(model, tokenizer, user_prompt='User:', alpha_2=0.5, alpha_3=0.5, len_opt=50, num_iterations=200, len_seq=200):
    
    import torch
    import tensorflow as tf
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import itertools
    from collections import defaultdict
    import numpy as np

    # Check if GPU is available
    if torch.cuda.is_available():
        # Use GPU
        device = next(model.parameters()).device
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Function to encode text using tokenizer
    def encode_text(texts, max_length=256, add_special_tokens=True, padding="max_length", truncation=True):
        encoding = tokenizer(texts, padding=padding, truncation=truncation, return_tensors="pt", max_length=max_length, add_special_tokens=False)
        return encoding.input_ids.to(device), encoding.attention_mask.to(device)

    # Get embedding matrix
    embedding_matrix = model.get_input_embeddings().weight.data

    # Perform prompt optimization
    for name, param in model.named_parameters():
        param.requires_grad = False

    def sentence_invert(user_prompt, alpha_2, alpha_3, len_opt, num_iterations, len_seq, max_length=256, learning_rate=0.1):
        # Vectorize the user prompt structure
        orgnl_input_ids, orgnl_attention_masks = encode_text([user_prompt], max_length=max_length, padding=False, truncation=True, add_special_tokens=False)
        # Get the input embedding vectors
        input_embeddings = model.get_input_embeddings()(orgnl_input_ids)
        # Create original target distribution
        orig_targ_distr = torch.zeros(orgnl_input_ids.shape[1], embedding_matrix.shape[0]).to(device)
        for targ_idx, targ_val in enumerate(orgnl_input_ids[0]):
            orig_targ_distr[targ_idx][targ_val] = 1

        # Initialize weight vector
        weight_vector = torch.rand(len_seq, embedding_matrix.shape[0], requires_grad=True, dtype=embedding_matrix.dtype).to(device)
        weight_vector.data.uniform_(0.0,1.0)
        # Detach the weight_vector tensor before passing it to the optimizer
        weight_vector = weight_vector.detach()
        weight_vector = weight_vector.requires_grad_(True)
        # Initialize optimizer
        optimizer = torch.optim.Adam([weight_vector], lr=learning_rate)

        global_weight_vector = None
        global_loss = 1e7
        visited_global_trigger = set()
        output_dict = defaultdict(lambda: 0)
        global_attn = None

        for i in range(1, num_iterations+1):
            optimizer.zero_grad()

            # Compute the weighted linear combination
            weighted_embedding = torch.matmul(weight_vector, embedding_matrix)
            weighted_embedding_expanded = weighted_embedding.unsqueeze(0)
            # Add weighted embedding to the user prompt structure
            input_embeddings_combined = torch.cat([input_embeddings[:, :, :], weighted_embedding_expanded[:, :, :]], dim=1)

            # Bypass the embedding layer and feed directly to the model
            outputs = model(inputs_embeds=input_embeddings_combined, output_attentions=True)

            # Perplexity loss
            output_probs_init = outputs.logits[:, :-1, :]
            output_probs_t = output_probs_init.permute(1, 0, 2)
            output_probs = output_probs_t.view(output_probs_t.shape[0]*output_probs_t.shape[1], output_probs_t.shape[2])
            target_distr = torch.cat([orig_targ_distr, weight_vector], dim=0)[1:,:]  
            spec_prob = torch.nn.functional.cross_entropy(output_probs, target_distr, reduction='mean')

            # Diversity loss
            # Calculate cosine similarity between each vector
            seq_similar = torch.nn.functional.cosine_similarity(weighted_embedding.unsqueeze(1), weighted_embedding.unsqueeze(0), dim=-1)
            # Find the mean cosine similarity
            diver_loss = torch.sum(seq_similar - torch.diag(torch.diag(seq_similar))) / (seq_similar.shape[0] * seq_similar.shape[1] - seq_similar.shape[0])

            # Attention loss            
            # Get the outputs of attention layers
            stacked_attns = torch.stack(outputs.attentions, dim=0) # layer, batch, head, token, :
            mean_attention = torch.mean(stacked_attns, dim=0)[0]
            normal_attn = (mean_attention - torch.mean(mean_attention, dim=0)) / (torch.std(mean_attention, dim=0) + 1e-7)
            attn_loss = torch.mean(torch.amax(normal_attn, dim=(0, 1)))

            # Define loss function
            loss = spec_prob + 100*alpha_2*torch.pow(diver_loss, 0.5) - alpha_3*attn_loss

            # Compute gradients and update weights
            loss.backward()
            gradients = weight_vector.grad

            # Forward pass
            # Find indices of the top negative gradients
            temp_top_vals, temp_indices = torch.topk(-gradients, k=1, sorted=True)
            temp_indices_comb = torch.cartesian_prod(*temp_indices)
            temp_vals_comb = list(itertools.product(*temp_top_vals))

            min_real_loss = 1e7
            local_weight_vector = None
            local_trigger = None
            local_attn = None
            for comb_idx, combination in enumerate(temp_indices_comb):
                # Set temp_weight_vector with 0s
                temp_weight_vector = torch.zeros_like(weight_vector)

                cur_temp_vals_comb = temp_vals_comb[comb_idx]
                sorted_indices_comb = sorted(range(len(cur_temp_vals_comb)), key=lambda cur_idx: cur_temp_vals_comb[cur_idx], reverse=True)
                temp_top_indices_comb = set(sorted_indices_comb[:len_opt])
                combination = tuple(torch.argmax(weight_vector[cur_idx,:]) if cur_idx not in temp_top_indices_comb else num for cur_idx, num in enumerate(combination))
                for index, idx_value in enumerate(combination):
                    # Set temp_weight_vector with 1s at the specified indices
                    temp_weight_vector[index][idx_value] = 1

                # Compute the weighted linear combination
                temp_weighted_embedding = torch.matmul(temp_weight_vector, embedding_matrix)
                temp_weighted_embedding_expanded = temp_weighted_embedding.unsqueeze(0)
                # Add weighted embedding to input embeddings
                temp_input_embeddings_combined = torch.cat([input_embeddings[:, :, :], temp_weighted_embedding_expanded[:, :, :]], dim=1)

                # Bypass the embedding layer and feed directly to the model
                temp_outputs = model(inputs_embeds=temp_input_embeddings_combined, output_attentions=True)

                # Perplexity loss
                temp_output_probs_init = temp_outputs.logits[:, :-1, :]
                temp_output_probs_t = temp_output_probs_init.permute(1, 0, 2)
                temp_output_probs = temp_output_probs_t.view(temp_output_probs_t.shape[0]*temp_output_probs_t.shape[1], temp_output_probs_t.shape[2])
                temp_target_distr = torch.cat([orig_targ_distr, temp_weight_vector], dim=0)[1:,:]
                temp_avg_spec_prob = torch.nn.functional.cross_entropy(temp_output_probs, temp_target_distr, reduction='mean')

                # Diversity loss
                # Calculate cosine similarity between each vector
                temp_embd = torch.index_select(embedding_matrix, 0, torch.tensor(combination, device=device))
                temp_seq_similar = torch.nn.functional.cosine_similarity(temp_embd.unsqueeze(1), temp_embd.unsqueeze(0), dim=-1)
                # Find the mean cosine similarity
                temp_diver_loss = torch.sum(temp_seq_similar - torch.diag(torch.diag(temp_seq_similar))) / (temp_seq_similar.shape[0] * temp_seq_similar.shape[1] - temp_seq_similar.shape[0])

                # Attention loss            
                # Get the outputs of attention layers
                temp_stacked_attns = torch.stack(temp_outputs.attentions, dim=0) # layer, batch, head, token, :
                temp_mean_attention = torch.mean(temp_stacked_attns, dim=0)[0]
                temp_normal_attn = (temp_mean_attention - torch.mean(temp_mean_attention, dim=0)) / (torch.std(temp_mean_attention, dim=0) + 1e-7)
                temp_attn_loss = torch.mean(torch.amax(temp_normal_attn, dim=(0, 1)))

                # Define loss function
                real_loss = temp_avg_spec_prob + 100*alpha_2*torch.pow(temp_diver_loss, 0.5) - alpha_3*temp_attn_loss

                # Update weight_vector
                if True: # real_loss < min_real_loss:
                    local_weight_vector = temp_weight_vector
                    local_trigger = tuple(token.item() for token in combination)
                    min_real_loss = real_loss
                    local_attn = temp_normal_attn
                    print(f'i: {i}')
                    print(f'Loss value: {real_loss}')
                    print(f'Sentence: {tokenizer.decode(combination, skip_special_tokens=False)}')

            output_dict[local_trigger] = min_real_loss
            weight_vector.data = local_weight_vector.data
            if min_real_loss < global_loss:
                global_weight_vector = local_weight_vector
                global_loss = min_real_loss
                global_attn = local_attn
                print(f'New best sentence: {tokenizer.decode(local_trigger, skip_special_tokens=False)}')

            if local_trigger in visited_global_trigger:
                return global_weight_vector, output_dict, global_attn
            else:
                visited_global_trigger.add(local_trigger)

        return global_weight_vector, output_dict, global_attn

    # Run the optimization
    optimized_weight_matrix = sentence_invert(user_prompt, alpha_2, alpha_3, len_opt, num_iterations, len_seq)

    best_sentence = optimized_weight_matrix[0].cpu().numpy()
    best_sent_tok = tf.nn.top_k(best_sentence, k=1, sorted=True)[1][:, 0].numpy()
    user_prompt_tok = tokenizer.encode(user_prompt, add_special_tokens=False)
    upd_best_sent_tok = np.concatenate((user_prompt_tok, best_sent_tok))
    interm_sents = optimized_weight_matrix[1]
    best_sent_attn = optimized_weight_matrix[2].cpu().numpy()

    return upd_best_sent_tok, interm_sents, best_sent_attn
