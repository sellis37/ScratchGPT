import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


# since we didn't really cover how to do this in lecture-
# this creates a learning rate schedule for you. Refer to the 
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step. 
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

# ===========================================================================

'''
Complete the following method which trains a GPT model and saves a loss curve.

To reiterate: you don't need to worry about weight decay, weight initialization, grad accumulation, or weight tying.
Use whatever batch size you are able, even something like 2 or 4 is fine.
Use a few hundred warmup steps and a peak learning rate that is (something x 10-4).
'''
def train():

    device = torch.device('cuda') # use 'cpu' if not gpu available

    # adjust as needed
    model = GPTModel(d_model=1024, n_heads=16, layers=32, vocab_size=10000, max_seq_len=512)
    param_count = sum(p.numel() for p in model.parameters())
    print('Model has', param_count, 'parameters.')

    model = model.to(device)

    # Load dataset
    data = np.load('packed_dataset.npy')
    data = torch.tensor(data, dtype=torch.long)

    batch_size = 16
    sequence_length = 512
    total_steps = 10000
    warmup_steps = 300

    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps)

    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    tokens_seen = 0

    model.train()

    for step in range(total_steps):

        # Sample random batch
        idx = torch.randint(0, data.shape[0], (batch_size,))
        batch = data[idx].to(device)

        # Shift inputs and targets
        inputs = batch[:, :-1]      # (B, S)
        targets = batch[:, 1:]      # (B, S)

        opt.zero_grad()

        logits = model(inputs)      # (B, S, V)

        # CrossEntropy expects (B, V, S)
        logits = logits.transpose(1, 2)

        loss = loss_fn(logits, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        scheduler.step()

        tokens_seen += batch_size * sequence_length
        losses.append(loss.item())

        if step % 50 == 0:
            print(f'Step {step} | Loss {loss.item():.4f}')

    # Plot loss curve
    token_axis = np.arange(len(losses)) * batch_size * sequence_length
    plt.plot(token_axis, losses)
    plt.xlabel('Tokens Processed')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('loss_curve.png')
    plt.close()

    # save model weights if you want
    torch.save(model.state_dict(), './model_weights.pt')

if __name__ == '__main__':
    train()