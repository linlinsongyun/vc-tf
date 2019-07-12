from .tacotron import Tacotron
from .nnet1 import NNet1

def create_model(name, hparams):
  if name == 'tacotron':
    return Tacotron(hparams)
  if name == 'nnet1':
    return NNet1(hparams)
  else:
    raise Exception('Unknown model: ' + name)
