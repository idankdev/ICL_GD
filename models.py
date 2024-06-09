import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.modules.container import ModuleList
from dataclasses import dataclass
from interventions import PermutationIntervention, LearningRateIntervention, SkipBlockIntervention


class AttnWrapper(torch.nn.Module):
    def __init__(self, attn, save_all=False):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None
        self.save_all = save_all

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        if self.save_all:
            self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, save_all=False):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.self_attn = self.block.self_attn = AttnWrapper(self.block.self_attn, save_all=save_all)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

        self.permutation = None
        self.save_all = save_all

        self.lr_scale = 1.0
        self.active = True


    def forward(self, *args, **kwargs):
        input = args[0]   # (batch, seq_length, d_model)
        if self.permutation is not None:
            # prev_input = input
            input = input[:, self.permutation, :]
            # assert not torch.all(torch.isclose(prev_input, input))

        output = self.block(input, *args[1:], **kwargs)
        if not self.active:
            output = (input,) + output[1:]
        if self.lr_scale != 1.0:
            # delta is the difference between the output and the input
            delta = output[0] - input
            output = (input + self.lr_scale * delta,) + output[1:]
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        if self.save_all:
            attn_output = self.block.self_attn.activations
            self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
            attn_output += input
            self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
            mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
            self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor
    
    def add_permutation(self, perm):
        self.permutation = perm
    
    def clean_permutation(self):
        self.permutation = None

    def reset(self):
        self.block.self_attn.reset()
        self.permutation = None
        self.lr_scale = 1.0
        self.active = True

    def get_attn_activations(self):
        return self.block.self_attn.activations

class Llama7BHelper:
    N_LAYERS = 32

    def __init__(self, model, device, token=None):
        '''
        model: "meta-llama/Llama-2-7b-hf" model
        '''
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token)
        self.model = model
        self.orig_model_layers = self.model.model.layers
        self.model.model.layers = ModuleList(
            [BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm) for layer in self.model.model.layers]
        )

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    def forward(self, prompt, interventions=[]):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'][0]
        for intrv in interventions:
            if isinstance(intrv, PermutationIntervention):
                assert intrv.perm.shape[0] == len(input_ids), f'Permutation index list of size {intrv.perm.shape[0]} but input sequence of length {len(input_ids)}'
                self.layers[intrv.before_layer_num].add_permutation(intrv.perm)
            
            elif isinstance(intrv, LearningRateIntervention):
                for layer in self.layers[intrv.start_layer_num:]:
                    layer.lr_scale = intrv.lr_scale
            
            elif isinstance(intrv, SkipBlockIntervention):
                self.layers[intrv.layer_num].active = False
            
            else:
                print(f'Intervention {intrv} not implemented')
                raise NotImplementedError
        with torch.no_grad():
            logits = self.model(inputs.input_ids.to(self.device)).logits
            # Cleanup
            for intrv in interventions:
                if isinstance(intrv, PermutationIntervention):
                    self.layers[intrv.before_layer_num].clean_permutation()

                if isinstance(intrv, LearningRateIntervention):
                    for layer in self.layers[intrv.start_layer_num:]:
                        layer.lr_scale = 1.0
            return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def get_prob_lens(self):
        '''
        Returns tensor of size (layer, pos, d_vocab) with probability distribution predictions for each token,
                in each layer. ASSUMES model.forward() has been run prior to function, and reset_all() has not been called
        '''
        layer_probs = []
        for layer in self.layers:
            inter_logits = layer.block_output_unembedded.squeeze(dim=0)   # pos, d_vocab
            inter_probs = torch.softmax(inter_logits, dim=-1)
            layer_probs.append(inter_probs)
        return torch.stack(layer_probs)

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))
    
    @property
    def layers(self):
        return self.model.model.layers
    
    @layers.setter
    def layers(self, layers):
        self.model.model.layers = layers



    def decode_all_layers(self, text, topk=10, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism', topk=topk)
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream', topk=topk)
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output', topk=topk)
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 'Block output', topk=topk)
    
    def reset_hf_model(self):
        self.model.model.layers = self.orig_model_layers