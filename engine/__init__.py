from .asr_transformer import Conformer
from .asr_seq2seq import CRDNN
from .asr_transducer import Conformer_Transducer
from .asr_ctc import ASR_CTC

class ASR_Controller:
    ctc = ASR_CTC
    seq2seq = CRDNN
    transformer = Conformer
    transducer = Conformer_Transducer
