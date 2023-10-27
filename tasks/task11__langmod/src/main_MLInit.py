# ## Taken from Karpathy's github: [url]

# import argparse
# import torch
# import torch.nn as nn
# from torch.nn import functional as F

# import data
# from model_cont import DTransformer
# from utils import get_batch, estimate_loss

# from interpolation import interpolate_weights

# parser = argparse.ArgumentParser()
# parser.add_argument('--debug', action='store_true')
# parser.add_argument('--num_layers', type=str, default='6')
# parser.add_argument('--num_epochs', type=str, default='5000')
# parser.add_argument('--scheme', type=str, default='Euler')
# args = parser.parse_args()

# def main():
#   # hyperparameters
#   batch_size = 64 # how many independent sequences will we process in parallel?
#   block_size = 256 # what is the maximum context length for predictions?
#   max_iters_list = [int(i) for i in args.num_epochs.split('-')]
#   eval_interval = 500
#   learning_rate = 3e-4
#   device = 'cuda' if torch.cuda.is_available() else 'cpu'
#   print(device)
#   eval_iters = 200
#   n_embd = 384
#   n_head = 6
#   n_layer_list = [int(i) for i in args.num_layers.split('-')]
#   dropout = 0.2
#   seed = 1337
#   # ------------

#   if args.debug:
#     batch_size = 2
#     block_size = 8
#     eval_interval = 1
#     eval_iters = 1
#     n_embd = 32
#     n_head = 4

#   torch.manual_seed(seed)

#   ## DATA
#   train_data, val_data, decode, vocab_size = data.main()

#   for k, (n_layer, max_iters) in enumerate(zip(n_layer_list, max_iters_list)):
#     ## MODEL
#     print(f'Building model w/ {n_layer} decoder layers')
#     model = DTransformer(n_embd, n_head, n_layer, dropout, vocab_size, 
#                                               block_size, args.scheme)
#     m = model.to(device)
#     # print the number of parameters in the model
#     print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#     # create a PyTorch optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#     if k != 0:
#       print(f'Interpolating weights from previous model to the new one')
#       interpolate_weights(model, old_model)

#     torch.manual_seed(seed)

#     print(f'Training model w/ {n_layer} decoder layers')
#     for iter in range(max_iters):
#         # every once in a while evaluate the loss on train and val sets
#         if iter % eval_interval == 0 or iter == max_iters - 1:
#             losses = estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device)
#             print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#         # sample a batch of data
#         xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)

#         # evaluate the loss
#         logits, loss = model(xb, yb)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#     # generate from the model
#     context = torch.zeros((1, 1), dtype=torch.long, device=device)
#     print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#     #open('../data/more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

#     old_model = model

# main()
