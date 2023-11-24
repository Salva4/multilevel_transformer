# import evaluate
import sacrebleu
import torch

# try:
#   metric = evaluate.load('sacrebleu')
# except:
#   print('sacrebleu failed. Loading bleu')
#   metric = evaluate.load('bleu')

def print_example(predictions, sources, targets_output, vocabulary_source, vocabulary_target, colour):
  question = sources[0,:]
  padding_question = (question == vocabulary_source.pad_id).nonzero()
  index_padding_question = padding_question[0] if len(
      padding_question != 0
  ) else len(question)
  text1 = ''.join(
      [
       vocabulary_source.id_to_string[i.item()] for i in question[
                                                      :index_padding_question]
      ]
  )

  answer_predicted = predictions[0,:]
  eos_answer_predicted = (
      answer_predicted == vocabulary_target.eos_id
  ).nonzero()
  index_eos_answer_predicted = eos_answer_predicted[0] if len(
      eos_answer_predicted != 0
  ) else len(answer_predicted)
  text2 = ''.join(
      [
       vocabulary_target.id_to_string[i.item()] for i in answer_predicted[
                                                :index_eos_answer_predicted+1]
      ]
  )

  answer_correct = targets_output[0,:]
  eos_answer_correct = (answer_correct == vocabulary_target.eos_id).nonzero()
  index_eos_answer_correct = eos_answer_correct[0] if len(
      eos_answer_correct != 0
  ) else len(answer_correct)
  text3 = ''.join(
      [
       vocabulary_target.id_to_string[i.item()] for i in answer_correct[
                                                  :index_eos_answer_correct+1]
      ]
  )
  text4 = bool(torch.logical_or(
      predictions == targets_output, 
      targets_output == vocabulary_target.pad_id
  ).prod(dim=1)[0].item())

  for pretext, text in [('QUESTION:', text1), 
                        ('PREDICTED ANSWER:', text2), 
                        ('CORRECT ANSWER:', text3), 
                        ('CORRECT?', 'Yes.' if text4 else 'No.')]:
    if colour_imported:
      print(colored(f'\t{pretext : >30} {text : <12}', colour))
    
    else:
      print(f'\t{pretext : >30} {text : <12}')

  print('\n\n')

def evaluate_bleu(_vars):
  _vars.model.eval() 
  with torch.no_grad():
    candidate_corpus, reference_corpus = [], []
    bleu = {'train': None, 'test': None}

    for mode in ['test']:#['train', 'test']:
      for i, instance in enumerate(_vars.dl[mode]):
        print(i)
        # if _vars.debug and i > 2: break
        src = instance['input_ids']
        translation = instance['translation'][_vars.lang_tgt]
        outputs = _vars.model.generate(
          src=src,
          max_new_tokens=40, 
          do_sample=False,#True, 
          top_k=30, 
          top_p=0.95,
          **_vars.__dict__,#
        )
        for j, output in enumerate(outputs):
          candidate = _vars.tokenizer.decode(
            output, 
            skip_special_tokens=True
          )
          reference = [translation[j]]
          candidate_corpus.append(candidate)
          reference_corpus.append(reference)

      bleu[mode] = sacrebleu.corpus_bleu(
        hypotheses=candidate_corpus, 
        references=reference_corpus,
      )

    _vars.candidate_corpus = candidate_corpus
    _vars.reference_corpus = reference_corpus
    _vars.bleu = bleu    




