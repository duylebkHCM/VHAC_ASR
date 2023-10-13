from .asr_transformer import Conformer
from .asr_seq2seq import CRDNN
from .asr_transducer import Conformer_Transducer
from .asr_ctc import ASR_CTC
from .asr_wav2vec2_ctc import ASR_WAV2VEC2_CTC
from .asr_wav2vec2_pretrain import ASR_WAV2VEC2_PRETRAIN
from .asr_wav2vec_transducer import Wav2vec_Transducer
from .asr_wav2vec2_hf_pretrain import HFW2VBrain
from .asr_hf_wav2vec_transducer import Wav2vecHF_Transducer

class ASR_Controller:
    wav2vec_transducer = Wav2vec_Transducer
    wav2vec = ASR_WAV2VEC2_PRETRAIN
    ctc = ASR_CTC
    wav2vec_ctc = ASR_WAV2VEC2_CTC
    seq2seq = CRDNN
    transformer = Conformer
    transducer = Conformer_Transducer
    hf_pretrain = HFW2VBrain
    hf_wav2vec_transducer = Wav2vecHF_Transducer