import time

def time_(f, *args, **kwargs):
  starting_time = time.time()
  output = f(*args, **kwargs)
  ending_time = time.time()

  duration_in_seconds = ending_time - starting_time

  return output, duration_in_seconds