


def run_cycle(model, optimizer, model_inputs, mu, nu, num_levels):
  for level in range(num_levels - 1):
    # loss = fwd_pass()
    model_outputs = model(**model_inputs)
    loss = model_outputs['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def MGOPT(model, optimizer, model_inputs, mu, nu, num_levels, num_iterations):
  for iteration in range(num_iterations):
    run_cycle(model, model_inputs, mu, nu, num_levels)

