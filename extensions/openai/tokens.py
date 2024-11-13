from extensions.openai.typing import DecodeRequest, EncodeRequest
from modules.text_generation import decode, encode


def token_count(request_data: EncodeRequest):
    tokens = encode(
        request_data.text,
        add_special_tokens=request_data.add_special_tokens,
        add_bos_token=request_data.add_bos_token,
        truncation_length=request_data.truncation_length
    )[0]
    return {
        'length': len(tokens)
    }


def token_encode(request_data: EncodeRequest):
    tokens = encode(
        request_data.text,
        add_special_tokens=request_data.add_special_tokens,
        add_bos_token=request_data.add_bos_token,
        truncation_length=request_data.truncation_length
    )[0]
    if tokens.__class__.__name__ in ['Tensor', 'ndarray']:
        tokens = tokens.tolist()

    return {
        'tokens': tokens,
        'length': len(tokens),
    }


def token_decode(request_data: DecodeRequest):
    output = decode(request_data.tokens, skip_special_tokens=request_data.skip_special_tokens)
    return {
        'text': output
    }
