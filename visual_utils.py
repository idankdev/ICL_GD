from matplotlib import pyplot as plt
import torch

def plot_icl_zsl_confidence(model, icl_prompt, zsl_prompt, correct_answer_str, icl_probs, zsl_probs):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Confidence in Final Answer Per Layer')
    for i, prompt in enumerate((icl_prompt, zsl_prompt)):
        print(prompt)
        prompt_type = 'ICL' if prompt == icl_prompt else 'ZSL'
        prob_lens = icl_probs if prompt == icl_prompt else zsl_probs
        final_answer_token = torch.argmax(prob_lens[-1, -1])
        final_answer_str = model.tokenizer.batch_decode(final_answer_token.unsqueeze(0))[0]
        print(f'Model Answer: {final_answer_str}, Correct Answer: {correct_answer_str}')
        correct = final_answer_str == correct_answer_str
        correct_str = "Correct" if correct else "Wrong"
        correct_answer_token = model.tokenizer.encode(text=correct_answer_str, add_special_tokens=False)[0]
        if correct:
            assert correct_answer_token == final_answer_token, (correct_answer_token, final_answer_token)
        

        final_answer_confidence = prob_lens[:, -1, final_answer_token].cpu()
        correct_answer_confidence = prob_lens[:, -1, correct_answer_token].cpu()
        
        for j in (0, 1):
            ax[i][j].plot(final_answer_confidence, label=f'Final ({correct_str}) Answer')
            if not correct:
                ax[i][j].plot(correct_answer_confidence, label='Correct Answer')
            ax[i][j].set_title(f'{"Log " if j == 1 else ''}Confidence - {prompt_type}')
            if j == 1:
                ax[i][j].semilogy()
            ax[i][j].legend()
    return fig, ax