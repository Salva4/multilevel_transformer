import torch.nn as nn

class PostContinuousBlock(nn.Module):
  def __init__(self, model_dimension, tokenizer, **kwargs):
    super().__init__()

    E_matrix = kwargs['model'].precontinuous_block.embedding.weight

    self.classifier = nn.Linear(model_dimension, len(tokenizer))
    # self.classifier.weight.data = E_matrix.data  # shared parameters

    ## Debugging enc-dec gradient_function
    # self.classifier = nn.Linear(model_dimension, model_dimension)

    # self.apply(init_weights)

  def forward(self, y, **kwargs):  # y: [b, L', d]
    logits = self.classifier(input=y)  # logits: [b, L', m]

    return {'x': logits}


