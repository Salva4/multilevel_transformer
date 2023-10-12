self.eval()
model.eval()

self = self.to(_vars.dev)
model = model.to(_vars.dev)

instance = next(iter(_vars.dl['train']))
src = instance['input_ids']

# torch.manual_seed(0)
# outputs_s = self.generate(
#   src,
#   max_new_tokens=40, 
#   do_sample=True, 
#   top_k=30, 
#   top_p=0.95
# )
# torch.manual_seed(0)
# outputs_m = model.generate(
#   src,
#   max_length=40, 
#   do_sample=True, 
#   top_k=30, 
#   top_p=0.95,
# )

input_ids = None
max_length = None
min_length = None
do_sample = None
early_stopping = None
num_beams = None
temperature = None
top_k = None
top_p = None
repetition_penalty = None
bad_words_ids = None
bos_token_id = None
pad_token_id = None
eos_token_id = None
length_penalty = None
no_repeat_ngram_size = None
num_return_sequences = None
attention_mask = None
decoder_start_token_id = None
use_cache = None

input_ids = src
max_new_tokens = 40
do_sample = True
top_k = 30
top_p = 0.95

self = model











