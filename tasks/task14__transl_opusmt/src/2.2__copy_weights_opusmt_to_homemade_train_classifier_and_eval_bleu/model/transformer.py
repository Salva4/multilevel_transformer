import copy
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

# from positional_encoding import PositionalEncoding
from model.transformer_encoder_layer import TransformerEncoderLayer
from model.transformer_decoder_layer import TransformerDecoderLayer
from model.transformer_utils import copy_weights, sequential
from model.beam_search_utils import BeamHypo

class Transformer(nn.Module):
  def __init__(self, _vars):
    super().__init__()

    self.d = _vars.d
    self.dev = _vars.dev
    self.pad_id = _vars.pad_id
    self.bos_id = _vars.bos_id
    self.eos_id = _vars.eos_id

    self.embedding = nn.Embedding(
      len(_vars.tokenizer), 
      _vars.d, 
      padding_idx=_vars.pad_id
    )

    # self.positional_encoding = PositionalEncoding(_vars.d)
    self.positional_encoding_src = nn.Embedding(512, 512)
    self.positional_encoding_tgt = nn.Embedding(512, 512)

    encoder_layer = TransformerEncoderLayer(_vars)
    decoder_layer = TransformerDecoderLayer(_vars)

    self.encoder = nn.ModuleList([
      copy.deepcopy(encoder_layer) for _ in range(_vars.num_layers_encoder)
    ])
    self.decoder = nn.ModuleList([
      copy.deepcopy(decoder_layer) for _ in range(_vars.num_layers_decoder)
    ])
    self.classifier = nn.Linear(_vars.d, len(_vars.tokenizer))

    self.encoder.forward = lambda x, *args, **kwargs: sequential(
      self.encoder, x, *args, **kwargs
    )
    self.decoder.forward = lambda x, *args, **kwargs: sequential(
      self.decoder, x, *args, **kwargs
    )

  def copy_weights(self, model):
    return copy_weights(self, model)

  def decode(self, tgt, mem, mask_pad_tgt, mask_pad_mem, 
                                  skip_embedding=False):
    ## Embedding and positional encoding
    if not skip_embedding:
      tgt = self.embed(x=tgt, whom='tgt')

    ## Decoder
    tgt = self.decoder(
      x=tgt, 
      memory=mem,
      mask_pad_tgt=mask_pad_tgt,
      mask_pad_mem=mask_pad_mem,
    )  # tgt: [b, L', d]

    return tgt

  def embed(self, x, whom):
    assert whom in ['src', 'tgt']

    ## Embedding
    x = self.embedding(x)  # src: [b, L , d]
                           # tgt: [b, L', d]
    
    ## Scale
    x *= np.sqrt(self.d)

    ## Positional encoding
    L = x.shape[1]
    positions = torch.arange(L).reshape(1, L).to(self.dev)  # positions_src: [1, L ]
                                                            # positions_tgt: [1, L']
    posenc_fn = self.positional_encoding_src if whom == 'src' else \
                self.positional_encoding_tgt
                
    posenc = posenc_fn(positions)  # positions_src: [1, L , d] 
                                   # positions_tgt: [1, L', d]

    # x = self.positional_encoding(x)  # src: [b, L , d]
                                       # tgt: [b, L', d]
    x += posenc  # src: [b, L , d]
                 # tgt: [b, L', d]
    return x

  def encode(self, src, mask_pad_src, skip_embedding=False):
    ## Embedding and positional encoding
    if not skip_embedding:
      src = self.embed(
        x=src, 
        whom='src'
      )

    ## Encoder
    mem = self.encoder(
      x=src,
      mask_pad_src=mask_pad_src
    )  # mem: [b, L, d]
    
    return mem

  def forward(self, src, tgt):  # src: [b, L ]
                                # tgt: [b, L']
    ## Padding masks for attention
    mask_pad_src = torch.where(src.eq(self.pad_id), -np.inf, 0)  # mask_pad_src: [b, L ]
    mask_pad_tgt = torch.where(tgt.eq(self.pad_id), -np.inf, 0)  # mask_pad_tgt: [b, L']
    mask_pad_mem = mask_pad_src          # mask_pad_mem: [b, L ]

    ## Encoder
    mem = self.encode(
      src=src, 
      mask_pad_src=mask_pad_src,
    )

    ## Decoder
    tgt = self.decode(
      tgt=tgt, 
      mem=mem,
      mask_pad_tgt=mask_pad_tgt,
      mask_pad_mem=mask_pad_mem,
    )

    ## Classifier
    logits = self.classifier(input=tgt)  # logits: [b, L', m]

    return logits

  def generate(self, src, **config_dict):
    config = SimpleNamespace(**config_dict)

    '''
        input_ids = src
        max_length = 512
        do_sample = True #False
        early_stopping = False
        num_beams = 4
        temperature = 1.
        top_k = 40 #50
        top_p = .95 #1.
        repetition_penalty = 1.
        bad_words_ids = [58100]  #[[58100]]
        bos_token_id = 0
        pad_token_id = 58100
        eos_token_id = 0
        length_penalty = 1.
        no_repeat_ngram_size = 0
        num_return_sequences = 1
        attention_mask = None
        decoder_start_token_id = 58100
        use_cache = True

        max_len = max_length
        len_penalty = length_penalty
        
        min_length = 0
        bad_words_ids = [[58100]]
    '''

    b, L, Lp, d = *src.shape, 1, self.d
    bp = num_beams * b

    ## Padding masks
    mask_pad_src = torch.where(src.eq(self.pad_id), -np.inf, 0)  # mask_pad_src: [b, L]
    mask_pad_tgt = None  # not needed

    ## Compute the memory once
    mem = self.encode(
      src=src, 
      mask_pad_src=mask_pad_src,
    )  # mem: [b, L, d]

    ## Expand mem & mask_pad_mem for each beam 
    mem = mem.reshape(b, 1, L, d).expand(b, num_beams, L, d).reshape(bp, L, d)  # mem: [b', L, d]
    mask_pad_mem = mask_pad_src.reshape(b, 1, L).expand(b, num_beams, L)\
                                                .reshape(bp, L)  # mask_pad_mem: [b', L]

    ## Init current state
    tgt = torch.full(
      size=(bp, 1), 
      fill_value=self.bos_id,
    ).long().to(src.device)
    
    if config.beam_search:
      hypos = [BeamHypo(num_beams, len_penalty) for _ in range(b)]
      beam_scores = torch.zeros((b, num_beams)).to(mem.device)  # beam_scores: [b, num_beams]
      done = [False] * b

      for curr_len in range(1, max_len):
        ## Decoder + classifier
        _tgt = self.decode(
          tgt=tgt, 
          mem=mem,
          mask_pad_tgt=mask_pad_tgt,
          mask_pad_mem=mask_pad_mem,
        )  # _tgt: [b', L', d]
        logits = self.classifier(input=_tgt)[:, -1, :]  # logits: [b', m]; b' = num_beams * b
        m = logits.shape[-1]

        score_logits = logits.log_softmax(-1)  # score_logits: [b', m]

        for banned_token_id in bad_words_ids:
          score_logits[:, banned_token_id] = -np.inf

        ## Add score_logits[i] to all the corresponding beams
        _scores = score_logits + beam_scores.reshape(-1, 1)  # _scores: [b', m]

        if do_sample:
          ## Top k
          ids2filter = (_scores < _scores.topk(top_k).values[:, -1].unsqueeze(-1))
          _scores[ids2filter] = -np.inf

          ## Top p
          if top_p < 1.:
            _scores_sorted, ids_sorted = _scores.sort(descending=True)  # _scores_sorted: [b', m]
                                                                      # ids_sorted: [b', m]
            cum_probs = _scores_sorted.softmax(-1).cumsum(-1)  # cum_probs: [b', m]
            ids_sorted_2filter = cum_probs > top_p  # ids_sorted_2filter: [b', m]
            ids_sorted_2filter[:, 1:] = ids_sorted_2filter[:, :-1].clone()
            ids_sorted_2filter[:, 0] = False

            ## make sure there are at least 2 non-filtered classes
            ids_sorted_2filter[:, :2] = False

            ids2filter = ids_sorted_2filter.scatter(-1, ids_sorted, ids_sorted_2filter)
            _scores[ids2filter] = -np.inf

          _scores = _scores.reshape(b, num_beams * m)  # _scores: [b, num_beams * m]
          probs = _scores.softmax(-1)  # probs: [b, num_beams * m]
          tokens = probs.multinomial(num_samples = 2*num_beams)  # tokens: [b, 2 * num_beams]
          scores = scores.gather(dim=-1, index=tokens)  # scores: [b, 2 * num_beams]
          scores, ids_new_scores = scores.sort(descending=True, dim=-1)  # scores: [b, 2 * num_beams]
          tokens = tokens.gather(-1, ids_new_scores)  # tokens: [b, 2 * num_beams]

        else:
          # raise Exception('Greedy decoding not implemented').
          scores = _scores.reshape(b, num_beams * m)
          scores, tokens = scores.topk(2 * num_beams)

        next_batch_beam = []

        for k in range(b):
          if done[k]:
            assert len(hypos[k]) >= num_beams
            next_batch_beam.extend(
              [(0, self.pad_id, 0)] * num_beams
            )
            continue

          next_sentence_beam = []

          for l in range(2 * num_beams):
            _token = tokens[k, l]
            score = scores[k, l]

            beam, token = _token // m, _token % m

            if token == self.eos_id:  # sentence finished
              if l >= num_beams:
                continue

              hypos[k].add(
                hypo=tgt[i*num_beams + beam].clone(), 
                sum_logprobs=score.item(),
              )

            else:
              next_sentence_beam.append(
                (score, token, l*num_beams + beam)
              )

            if len(next_sentence_beam) == num_beams: 
              break

          done[k] = done[k] or hypos[k].is_done(
            best_sum_logprobs=scores[k].max().item(), 
            curr_len=curr_len,
          )

          next_batch_beam.extend(next_sentence_beam)

        if all(done):
          break

        assert len(next_batch_beam) == bp
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])  # beam_scores: [bp]
        tokens      =         src.new([x[1] for x in next_batch_beam])  #      tokens: [bp]
        ids         =         src.new([x[2] for x in next_batch_beam])  #         ids: [bp]

        tgt = tgt[ids]
        tgt = torch.cat(
          (tgt, tokens.unsqueeze(-1)),
          axis=-1
        )
        curr_len += 1

      for k in range(b):
        if done[k]:
          continue

        for i in range(num_beams):
          score = beam_scores[num_beams * k + i]
          hypo = tgt[num_beams * k + i]
          hypos[k].add(
            hypo=hypo,
            sum_logprobs=score.item(),
          )

      # ...

      sentence_lens = tgt.new(b)
      best = []

      for k, hypo in enumerate(hypos):
        beams_sorted = sorted(hypo.beams)
        best_beam = beams_sorted.pop()
        sentence_lens[k] = len(best_beam)
        best.append(best_beam)

      Lmax = min(sentence_lens.max() + 1, max_len)
      generated = tgt.new(b, Lmax).fill_(self.pad_id)

      for k, hypo in enumerate(best):
        generated[k, :sentence_lens[k]] = hypo
        generated[k,  sentence_lens[k]] = self.eos_id

      return generated

    else:
      raise Exception('No-beam_search not implemented.')



























