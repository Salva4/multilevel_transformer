


def copy_weights(pretrained_model, local_model):
  ## Embedding
  local_model.precontinuous_block.embedding.weight.data = \
    pretrained_model.model.shared.weight.data

  ## Positional encoding
  local_model.precontinuous_block.positional_encoding_src.weight.data = \
    pretrained_model.model.encoder.embed_positions.weight.data
  local_model.precontinuous_block.positional_encoding_tgt.weight.data = \
    pretrained_model.model.decoder.embed_positions.weight.data

  assert local_model.continuous_blocks[0].num_layers \
      == len(pretrained_model.model.encoder.layers)
  ## Encoder
  for i in range(len(pretrained_model.model.encoder.layers)):
    ## Self-attention
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.k_proj.weight.data = \
      pretrained_model.model.encoder.layers[i].self_attn.k_proj.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.k_proj.bias.data = \
      pretrained_model.model.encoder.layers[i].self_attn.k_proj.bias.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.v_proj.weight.data = \
      pretrained_model.model.encoder.layers[i].self_attn.v_proj.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.v_proj.bias.data = \
      pretrained_model.model.encoder.layers[i].self_attn.v_proj.bias.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.q_proj.weight.data = \
      pretrained_model.model.encoder.layers[i].self_attn.q_proj.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.q_proj.bias.data = \
      pretrained_model.model.encoder.layers[i].self_attn.q_proj.bias.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.out_proj.weight.data = \
      pretrained_model.model.encoder.layers[i].self_attn.out_proj.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn.attn.out_proj.bias.data = \
      pretrained_model.model.encoder.layers[i].self_attn.out_proj.bias.data

    ## MLP
    local_model.continuous_blocks[0].layers[i].residual_layer.F.mlp.fc1.weight.data = \
      pretrained_model.model.encoder.layers[i].fc1.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.mlp.fc1.bias.data = \
      pretrained_model.model.encoder.layers[i].fc1.bias.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.mlp.fc2.weight.data = \
      pretrained_model.model.encoder.layers[i].fc2.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.mlp.fc2.bias.data = \
      pretrained_model.model.encoder.layers[i].fc2.bias.data

    ## Layer normalization
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn_layer_norm.weight.data = \
      pretrained_model.model.encoder.layers[i].self_attn_layer_norm.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.self_attn_layer_norm.bias.data = \
      pretrained_model.model.encoder.layers[i].self_attn_layer_norm.bias.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.mlp_layer_norm.weight.data = \
      pretrained_model.model.encoder.layers[i].final_layer_norm.weight.data
    local_model.continuous_blocks[0].layers[i].residual_layer.F.mlp_layer_norm.bias.data = \
      pretrained_model.model.encoder.layers[i].final_layer_norm.bias.data

  assert local_model.continuous_blocks[0].num_layers \
      == len(pretrained_model.model.decoder.layers)
  ## Decoder
  for i in range(len(pretrained_model.model.decoder.layers)):
    ## Self-attention
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.k_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].self_attn.k_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.k_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].self_attn.k_proj.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.v_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].self_attn.v_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.v_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].self_attn.v_proj.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.q_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].self_attn.q_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.q_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].self_attn.q_proj.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.out_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].self_attn.out_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn.attn.out_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].self_attn.out_proj.bias.data

    ## Cross-attention
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.k_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.k_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.k_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.k_proj.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.v_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.v_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.v_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.v_proj.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.q_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.q_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.q_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.q_proj.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.out_proj.weight.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.out_proj.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn.attn.out_proj.bias.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn.out_proj.bias.data      

    ## MLP
    local_model.continuous_blocks[1].layers[i].residual_layer.F.mlp.fc1.weight.data = \
      pretrained_model.model.decoder.layers[i].fc1.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.mlp.fc1.bias.data = \
      pretrained_model.model.decoder.layers[i].fc1.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.mlp.fc2.weight.data = \
      pretrained_model.model.decoder.layers[i].fc2.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.mlp.fc2.bias.data = \
      pretrained_model.model.decoder.layers[i].fc2.bias.data

    ## Layer normalization
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn_layer_norm.weight.data = \
      pretrained_model.model.decoder.layers[i].self_attn_layer_norm.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.self_attn_layer_norm.bias.data = \
      pretrained_model.model.decoder.layers[i].self_attn_layer_norm.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn_layer_norm.weight.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn_layer_norm.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.cross_attn_layer_norm.bias.data = \
      pretrained_model.model.decoder.layers[i].encoder_attn_layer_norm.bias.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.mlp_layer_norm.weight.data = \
      pretrained_model.model.decoder.layers[i].final_layer_norm.weight.data
    local_model.continuous_blocks[1].layers[i].residual_layer.F.mlp_layer_norm.bias.data = \
      pretrained_model.model.decoder.layers[i].final_layer_norm.bias.data

  ## Classifier
  local_model.postcontinuous_block.classifier.weight.data = \
    pretrained_model.lm_head.weight.data
  local_model.postcontinuous_block.classifier.bias.data = \
    pretrained_model.final_logits_bias.reshape(
      local_model.postcontinuous_block.classifier.bias.data.shape
    )

def sequential(model, x, *args, **kwargs):  
  for layer in model:
    x = layer(x, *args, **kwargs)

  return x




