"""Defines the model types.
"""

from model.debug_model import DebugModel
from model.fusion_net import FusionNet
from model.match_lstm import MatchLstm
from model.mnemonic_reader import MnemonicReader
from model.qa_model import QaModel
from model.rnet import Rnet

MODEL_TYPES = {
    "debug": DebugModel,
    "fusion_net": FusionNet,
    "match_lstm": MatchLstm,
    "mnemonic_reader": MnemonicReader,
    "qa_model": QaModel,
    "rnet": Rnet,
}
