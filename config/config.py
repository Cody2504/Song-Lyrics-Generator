import torch
class Config:
    MAX_SEQ_LEN = 128
    SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>', '<eol>']

    EMBEDDING_DIMS = 128
    HIDDEN_DIMS = 128
    N_LAYERS = 2
    N_HEADS = 4
    DROPOUT = 0.2

    BATCH_SIZE = 64
    LEARNING_RATE = 3.0
    NUM_EPOCHS = 100
    GRAD_CLIP = 0.5

    MAX_GENERATION_LENGTH = 128
    TEMPERATURE = 1.2
    DEFAULT_NUM_SEGMENTS = 3

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'