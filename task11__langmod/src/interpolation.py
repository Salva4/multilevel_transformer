def interpolate_weights(model, old_model):
  ## Copy weights Embedding and Classifier
  for (p_m, p_om) in zip(model.emb.parameters(),
                     old_model.emb.parameters()):
    p_m.data = p_om.data
  for (p_m, p_om) in zip(model.fc.parameters(),
                     old_model.fc.parameters()):
    p_m.data = p_om.data

  ## Constant interpolation: Decoder
  for l in range(old_model.decoder.N):
    for (p_om, p_m_1, p_m_2) in zip(old_model.decoder.phi[l].parameters(),
                                      model.decoder.phi[2*l].parameters(),
                                   model.decoder.phi[2*l+1].parameters()):
      p_m_1.data = p_om.data
      p_m_2.data = p_om.data
































