## Case I: Usual transformer
# def interpolate_weights(model, old_model):
#   ## Copy weights Embedding and Classifier
#   for (p_m, p_om) in zip(model.embedding_encoder.parameters(),
#                      old_model.embedding_encoder.parameters()):
#     p_m.data = p_om.data
#   for (p_m, p_om) in zip(model.embedding_decoder.parameters(),
#                      old_model.embedding_decoder.parameters()):
#     p_m.data = p_om.data
#   for (p_m, p_om) in zip(model.fc.parameters(),
#                      old_model.fc.parameters()):
#     p_m.data = p_om.data

#   ## Linear interpolation (except Constant for last)
#   for l in range(old_model.encoder.num_layers):
#     for (param_model, param_model2) in zip(
#                                  old_model.encoder.layers[l].parameters(), 
#                                   model.encoder.layers[2*l].parameters()):
#       param_model2.data = param_model.data
#   for l in range(old_model.encoder.num_layers - 1):
#     for (param_l, param_lp1, param_lp2) in zip(
#                                    model.encoder.layers[2*l].parameters(),
#                                  model.encoder.layers[2*l+1].parameters(),
#                                 model.encoder.layers[2*l+2].parameters()):
#       # param_lp1.data = (param_l.data + param_lp2.data)/2
#       param_lp1.data = param_l.data
#   l = old_model.encoder.num_layers - 1
#   for (param_l, param_lp1) in zip(
#                                  model.encoder.layers[2*l].parameters(),
#                               model.encoder.layers[2*l+1].parameters()):
#     param_lp1.data = param_l.data

## Case II: Transformer with an EulerContUnit
def interpolate_weights(old_model, new_model, interpolation='constant'):
  ## Copy weights Embedding and Classifier
  for (p_nm, p_om) in zip(new_model.embedding_encoder.parameters(),
                          old_model.embedding_encoder.parameters()):
    p_nm.data = p_om.data
  for (p_nm, p_om) in zip(new_model.embedding_decoder.parameters(),
                          old_model.embedding_decoder.parameters()):
    p_nm.data = p_om.data
  for (p_nm, p_om) in zip(new_model.fc.parameters(),
                          old_model.fc.parameters()):
    p_nm.data = p_om.data

  ## Encoder and Decoder: constant/linear interpolation
  ## (I) Encoder
  for l in range(old_model.encoder.N):
    for (p_om_1, p_om_2, p_nm_1, p_nm_2) in zip(
                                   old_model.encoder.phi[  l  ].parameters(),
                                   old_model.encoder.phi[  l+1].parameters(),
                                   new_model.encoder.phi[2*l  ].parameters(),
                                   new_model.encoder.phi[2*l+1].parameters()):
      if interpolation == 'constant':
        p_nm_1.data = p_om_1.data
        p_nm_2.data = p_om_1.data
      elif interpolation == 'linear':
        p_nm_1.data = p_om_1.data
        p_nm_2.data = (p_om_1.data + p_om_2.data)/2
  ## ...copy last layer 
  for (p_om, p_nm) in zip(old_model.encoder.phi[old_model.encoder.N].parameters(),
                          new_model.encoder.phi[new_model.encoder.N].parameters()):
    p_nm.data = p_om.data

  ## (II) Decoder
  for l in range(old_model.decoder.N):
    for (p_om_1, p_om_2, p_nm_1, p_nm_2) in zip(
                                   old_model.decoder.phi[  l  ].parameters(),
                                   old_model.decoder.phi[  l+1].parameters(),
                                   new_model.decoder.phi[2*l  ].parameters(),
                                   new_model.decoder.phi[2*l+1].parameters()):
      if interpolation == 'constant':
        p_nm_1.data = p_om_1.data
        p_nm_2.data = p_om_1.data
      elif interpolation == 'linear':
        p_nm_1.data = p_om_1.data
        p_nm_2.data = (p_om_1.data + p_om_2.data)/2
  ## ...copy last layer 
  for (p_om, p_nm) in zip(old_model.decoder.phi[old_model.decoder.N].parameters(),
                          new_model.decoder.phi[new_model.decoder.N].parameters()):
    p_nm.data = p_om.data

## Moved to R_I_MGOpt.py ########################
# def restrict_weights(old_model, new_model):
#   ## Copy weights Embedding and Classifier
#   for (p_om, p_nm) in zip(old_model.embedding_encoder.parameters(),
#                           new_model.embedding_encoder.parameters()):
#     p_nm.data = p_om.data
#   for (p_om, p_nm) in zip(old_model.embedding_decoder.parameters(),
#                           new_model.embedding_decoder.parameters()):
#     p_nm.data = p_om.data
#   for (p_om, p_nm) in zip(old_model.fc.parameters(),
#                           new_model.fc.parameters()):
#     p_nm.data = p_om.data

#   ## Encoder and Decoder: restriction by averaging even & odd
#   ## (I) Encoder
#   for l in range(new_model.encoder.N):
#     for (p_om_1, p_om_2, p_nm) in zip(old_model.encoder.phi[2*l].parameters(),
#                                     old_model.encoder.phi[2*l+1].parameters(),
#                                        new_model.encoder.phi[l].parameters()):
#       p_nm.data = (p_om_1.data + p_om_2.data)/2
#   ## ...copy last layer 
#   for (p_om, p_nm) in zip(old_model.encoder.phi[old_model.encoder.N].parameters(),
#                           new_model.encoder.phi[new_model.encoder.N].parameters()):
#     p_nm.data = p_om.data

#   ## (II) Decoder
#   for l in range(new_model.decoder.N):
#     for (p_om_1, p_om_2, p_nm) in zip(old_model.decoder.phi[2*l].parameters(),
#                                     old_model.decoder.phi[2*l+1].parameters(),
#                                        new_model.decoder.phi[l].parameters()):
#       p_nm.data = (p_om_1.data + p_om_2.data)/2
#   ## ...copy last layer 
#   for (p_om, p_nm) in zip(old_model.encoder.phi[old_model.encoder.N].parameters(),
#                           new_model.encoder.phi[new_model.encoder.N].parameters()):
#     p_nm.data = p_om.data


# def restrict_grads(old_model, new_model): <-- was this even used? I think not
#   ## Copy grads Embedding and Classifier
#   for (p_om, p_nm) in zip(old_model.embedding_encoder.parameters(),
#                           new_model.embedding_encoder.parameters()):
#     p_nm.data = p_om.data
#   for (p_om, p_nm) in zip(old_model.embedding_decoder.parameters(),
#                           new_model.embedding_decoder.parameters()):
#     p_nm.data = p_om.data
#   for (p_om, p_nm) in zip(old_model.fc.parameters(),
#                           new_model.fc.parameters()):
#     p_nm.data = p_om.data

#   ## Encoder and Decoder: restriction by averaging even & odd
#   ## (I) Encoder
#   for l in range(new_model.encoder.N):
#     for (p_om_1, p_om_2, p_nm) in zip(old_model.encoder.phi[2*l].parameters(),
#                                     old_model.encoder.phi[2*l+1].parameters(),
#                                        new_model.encoder.phi[l].parameters()):
#       p_nm.data = (p_om_1.data + p_om_2.data)/2
#   ## ...copy last layer 
#   for (p_om, p_nm) in zip(old_model.encoder.phi[old_model.encoder.N].parameters(),
#                           new_model.encoder.phi[new_model.encoder.N].parameters()):
#     p_nm.data = p_om.data

#   ## (II) Decoder
#   for l in range(new_model.decoder.N):
#     for (p_om_1, p_om_2, p_nm) in zip(old_model.decoder.phi[2*l].parameters(),
#                                     old_model.decoder.phi[2*l+1].parameters(),
#                                        new_model.decoder.phi[l].parameters()):
#       p_nm.data = (p_om_1.data + p_om_2.data)/2
#   ## ...copy last layer 
#   for (p_om, p_nm) in zip(old_model.encoder.phi[old_model.encoder.N].parameters(),
#                           new_model.encoder.phi[new_model.encoder.N].parameters()):
#     p_nm.data = p_om.data
#################################################
































