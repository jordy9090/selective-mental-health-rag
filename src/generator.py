import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel


def load_model(base_id, adapter_dir=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
    )

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return model, tok


@torch.no_grad()
def generate(
    model,
    tok,
    system,
    user,
    max_new=400,
    do_sample=True,
    temperature=0.7,
    top_p=1.0,
):
    """
    일반 답변용:
        generate(..., do_sample=True, temperature=0.7, top_p=1.0)

    gate scoring용:
        generate(..., do_sample=False)

    참고:
    - do_sample=False면 모델이 매 step마다 가장 확률 높은 토큰을 고름
    - 이때 temperature/top_p는 의미가 거의 없어서 코드상 아예 넘기지 않는 게 깔끔함
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user}],
        },
    ]

    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
    )

    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        gen_kwargs.update(
            do_sample=False,
        )

    out = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = out[0][prompt_len:]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()
