class Config:
    def __init__(self) -> None:
        self.max_len = 100
        self.learning_rate = 0.001
        self.weight_decay = 0.1
        self.blstm_input_size = 128
        self.blstm_hidden_size = 256
        self.blstm_num_layers = 1
        self.mlp_hidden_size = 256
        self.embedding_size = 128