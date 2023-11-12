import torch.nn as nn

class PostContinuousBlock(nn.Module):
  def __init__(self, model_dimension, tokenizer, **kwargs):
    super().__init__()

    self.classifier = nn.Linear(model_dimension, len(tokenizer))

    # self.apply(init_weights)

  def forward(self, y, **kwargs):  # y: [b, L', d]
    logits = self.classifier(input=y)  # logits: [b, L', m]

    return {'x': logits}


