
import sys
sys.path.append('../../../src/')

from pipeline.pipeline_template import main as run_pipeline
from argument_parsing import parse_arguments, assert_and_correct_arguments
from data import obtain_data

print('Parsing arguments...')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.\n')
print(f'Args: {args}')

## debug mode #######################
# if args.debug:
#   args.batch_size = 2
#   # args.continuous = True
#   args.max_length = 10
#   args.num_layers = 8#2
#   args.T = float(args.num_layers)
#####################################

def get_batch_fn_generator(_vars):
  _vars.data_loader_iterators = dict(zip(
    _vars.splits, [iter(_vars.data_loaders[split]) for split in _vars.splits],
  ))
  
  def get_batch(split):
    batch = next(_vars.data_loader_iterators[split], None)

    if batch is None:
      _vars.data_loader_iterators[split] = iter(_vars.data_loaders[split])
      batch = next(_vars.data_loader_iterators[split], None)
      if batch is None: 
        raise Exception(f'Length of {split} data loader is 0.')

    input, target = batch
    batch = (input, target)

    return batch

  return get_batch

def main():
  continuous_blocks_num_layers = [args.num_layers]
  continuous_blocks_T = [args.T]

  run_pipeline(
    args=args,
    obtain_data=obtain_data,
    continuous_blocks_num_layers=continuous_blocks_num_layers,
    continuous_blocks_T=continuous_blocks_T,
    get_batch_fn_generator=get_batch_fn_generator,
    compute_accuracy=True,
    print_times=False,
  )

if __name__ == '__main__': main()




