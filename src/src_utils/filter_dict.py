# import inspect

# def filter_keys(dictionary, function):
#   dictionary = dictionary.copy()
#   parameters = list(inspect.signature(function).parameters)

#   for parameter in parameters: 
#     if parameter in dictionary.keys(): 
#       _ = dictionary.pop(parameter)

#   return dictionary

def filter_keys(dictionary, already_provided_parameters):
  dictionary = dictionary.copy()

  for parameter in already_provided_parameters: 
    if parameter in dictionary.keys(): 
      _ = dictionary.pop(parameter)

  return dictionary




