from typing import Optional
from llama import Llama
import json
import torch


import fire


class MyHook:
    def __init__(self):
        self.should_transform = lambda layer_id, layer_name: False
        self.transform = lambda layer_id, layer_name, x: x

    def post(self, layer_id, layer_name, x):
        if self.should_transform(layer_id, layer_name):
            x = self.transform(layer_id, layer_name, x)
        return x


def main(
    ckpt_dir: str = "/workspace/llama-2-7b-chat/",
    tokenizer_path: str = "/workspace/llama-2-7b-chat/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    hook = MyHook()
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        hook=hook,
    )

    dialogs = [
        [{"role": "user", "content": "How do I make ketchup?"}],
        [
            {
                "role": "system",
                "content": "Respond accurately and concisely without patronizing",
            },
            {"role": "user", "content": "How do I make ketchup?"},
        ],
        [{"role": "user", "content": "Who is the president of the United States?"}],
        [
            {
                "role": "system",
                "content": "Respond accurately and concisely without patronizing",
            },
            {"role": "user", "content": "Who is the president of the United States?"},
        ],
    ]

    # mask = (torch.rand((1, 1, 4096)) > 0.6).cuda()

    # def should_transform(layer_id, layer_name):
    #     # return layer_name in ["attn", "ffn"] and layer_id == 2
    #     return layer_name in ["attn", "ffn"] and layer_id % 2 == 1
    #     # return False

    # def transform(layer_id, layer_name, x):
    #     return x * mask
    #     # return torch.zeros_like(x)

    # hook.should_transform = should_transform
    # hook.transform = transform

    all_results = []
    for lid in range(-1, generator.model.params.n_layers):
        hook.should_transform = lambda layer_id, layer_name: (layer_id == lid) and (
            layer_name in ["attn", "ffn"]
        )
        hook.transform = lambda layer_id, layer_name, x: torch.zeros_like(x)

        torch.manual_seed(1)
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        all_results.append(results)

        print(f"-------- Layer {lid} results: ---------")

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

    with open("results.json", "w") as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    fire.Fire(main)
