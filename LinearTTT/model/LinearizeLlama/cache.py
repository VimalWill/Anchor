import inspect

from transformers.cache_utils import Cache


class TTTCache(Cache):
    """Per-request fast weights, pending chunk updates, and local attention KV."""

    def __init__(self):
        # Newer HF Cache requires a layers arg; TTT state lives in self.states.
        if 'layers' in inspect.signature(Cache.__init__).parameters:
            super().__init__(layers=[])
        else:
            super().__init__()
        self.states = {}
        self._seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens

    def get_max_length(self):
        return None

    def update(self, *args, **kwargs):
        raise NotImplementedError('TTTCache is updated by the TTT decoder.')

    def reorder_cache(self, beam_idx):
        raise NotImplementedError('TTTCache supports greedy/sampling decode, not beam search.')

    def to_legacy_cache(self):
        raise NotImplementedError('A KV-only legacy cache cannot represent TTT fast weights.')
