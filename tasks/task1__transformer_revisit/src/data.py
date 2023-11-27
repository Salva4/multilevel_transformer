import input_pipeline
import preprocessing

TRAINING_DATA_PATH = '../data/en_gum-ud-train.conllu.txt'#'/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
VALIDATION_DATA_PATH = '../data/en_gum-ud-dev.conllu.txt'#'/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
TRAINING_DATA_PATH_DEBUG = '../data/en_gum-ud-train.conllu_debug.txt'
VALIDATION_DATA_PATH_DEBUG = '../data/en_gum-ud-dev.conllu_debug.txt'

def obtain_data(_vars):
  training_data_path   =   TRAINING_DATA_PATH if not _vars.debug else \
                           TRAINING_DATA_PATH_DEBUG
  validation_data_path = VALIDATION_DATA_PATH if not _vars.debug else \
                         VALIDATION_DATA_PATH_DEBUG
  print(
    'training_data_path'  ,   training_data_path,
    'validation_data_path', validation_data_path,
  )

  vocabularies = input_pipeline.create_vocabularies(training_data_path)
  vocabulary_size = len(vocabularies['forms'])
  num_classes = len(vocabularies['xpos'])

  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]

  training_data_set, training_data_loader = preprocessing.obtain_dataset(
    filename=training_data_path, 
    vocabularies=vocabularies, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=_vars.batch_size, 
    bucket_size=_vars.max_length,
    seed=0,
  )
  validation_data_set, validation_data_loader = preprocessing.obtain_dataset(
    filename=validation_data_path, 
    vocabularies=vocabularies, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=_vars.batch_size,#187, 
    bucket_size=_vars.max_length,
    seed=0,
  )

  _vars.splits = ['training', 'validation']
  _vars.data_sets = {
      'training':   training_data_set,
    'validation': validation_data_set,
  }
  _vars.data_loaders = {
      'training':   training_data_loader,
    'validation': validation_data_loader,
  }
  _vars.vocabulary_size = vocabulary_size
  _vars.num_classes = num_classes
  _vars.pad_token_id = _vars.data_sets['training'].pad_token_id

  return




