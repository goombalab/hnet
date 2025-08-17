from dataclasses import dataclass

from numpy import array, ndarray, uint8


@dataclass
class ByteTokenizer:
    vocab_size = 256
    bos_idx = 254
    eos_idx = 255
    dtype = uint8

    def encode(self, text: str, add_bos=False, add_eos=False) -> ndarray:
        u8s = text.encode("utf-8")
        if add_bos:
            u8s = bytes([self.bos_idx]) + u8s
        if add_eos:
            u8s = u8s + bytes([self.eos_idx])
        u8s = bytearray(u8s)
        tokens = array(u8s, dtype=self.dtype)
        return tokens

    def decode(self, tokens):
        return bytearray(tokens).decode("utf-8")
