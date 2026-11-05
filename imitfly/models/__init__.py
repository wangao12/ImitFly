from .policy import ImitFlyPolicy

def make_policy(name="imitfly", sequence_length=5):
    if name == "imitfly":
        return ImitFlyPolicy(sequence_length)
    if name == "no_scve":
        return ImitFlyPolicy(sequence_length, use_spiking=False)
    if name == "no_hmam":
        return ImitFlyPolicy(sequence_length, use_attention=False)
    if name == "no_lstm":
        return ImitFlyPolicy(sequence_length, use_memory=False)
    if name == "walking":
        from .walking import WalkingPolicy
        return WalkingPolicy()
    if name == "feedforward":
        from .feedforward import FeedForwardPolicy
        return FeedForwardPolicy()
    if name == "lstm":
        from .lstm import LSTMPolicy
        return LSTMPolicy(sequence_length)
    if name == "lstm_attention":
        from .lstm_attention import LSTMAttentionPolicy
        return LSTMAttentionPolicy(sequence_length)
    raise ValueError(f"Unknown policy: {name}")
