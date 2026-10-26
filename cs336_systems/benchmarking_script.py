from cs336_basics import model, optimizer, nn_utils
import torch
import timeit
import numpy as np
import pandas as pd


def generate_random_batch_data(vocab_size, context_length, batch_size, device):

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    return x, y


def get_hyperparameter_grid():
    d = {
        # "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        # "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        # "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
        # "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
    }
    return d


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def main():
    records = {}
    out = {}

    vocab_size = 10_000
    context_length = 512
    rope_theta = 10_000.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    start_it = 0
    max_iterations = 15
    max_learning_rate = 1e-3
    min_learning_rate = 3e-5
    warmup_iters = 5
    cosine_cycle_iters = 10
    forward_pass = True
    backward_pass = True
    optimizer_pass = True

    hyperparameter_grid = get_hyperparameter_grid()
    for key, value in hyperparameter_grid.items():
        name = key
        tmp = []

        d_model = value["d_model"]
        d_ff = value["d_ff"]
        num_layers = value["num_layers"]
        num_heads = value["num_heads"]
        transformer_model = model.BasicsTransformerLM(
            vocab_size=vocab_size, context_length=context_length, d_model=d_model, d_ff=d_ff, num_layers=num_layers, num_heads=num_heads, rope_theta=rope_theta
        )
        x, y = generate_random_batch_data(vocab_size, context_length, batch_size, device)
        for it in range(start_it, max_iterations):
            opt = optimizer.AdamW(transformer_model.parameters())
            lr = optimizer.get_cosine_lr(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
            set_optimizer_lr(opt, lr)
            transformer_model.train()
            start = timeit.default_timer()
            opt.zero_grad()
            preds = transformer_model(x)
            forward_pass_elapsed = timeit.default_timer() - start
            # print(y)
            loss = nn_utils.cross_entropy(preds, y)
            loss.backward()
            backward_pass_elapsed = timeit.default_timer() - start - forward_pass_elapsed
            opt.step()
            optimizer_elapsed = timeit.default_timer() - start - forward_pass_elapsed - backward_pass_elapsed
            if it > warmup_iters - 1:
                record = {"it": it + 1}
                if forward_pass:
                    record["forward_pass_elapsed"] = forward_pass_elapsed
                if backward_pass:
                    record["backward_pass_elapsed"] = backward_pass_elapsed
                if optimizer_pass:
                    record["optimizer_elapsed"] = optimizer_elapsed
                tmp.append(record)
        records[name] = tmp
    # print(records)
    for key, value in records.items():
        forward_pass_elapsed_tmp, backward_pass_elapsed_tmp, foptimizer_elapsed_tmp = [], [], []
        for record in value:
            # print(record)
            forward_pass_elapsed_tmp.append(record["forward_pass_elapsed"])
            backward_pass_elapsed_tmp.append(record["backward_pass_elapsed"])
            foptimizer_elapsed_tmp.append(record["optimizer_elapsed"])
        out[key] = {
            "avg_forward_pass_elapsed": np.mean(forward_pass_elapsed_tmp),
            "std_forward_pass_elapsed": np.std(forward_pass_elapsed_tmp),
            "avg_backward_pass_elapsed": np.mean(backward_pass_elapsed_tmp),
            "std_backward_pass_elapsed": np.std(backward_pass_elapsed_tmp),
            "avg_optimizer_elapsed": np.mean(foptimizer_elapsed_tmp),
            "std_optimizer_elapsed": np.std(foptimizer_elapsed_tmp),
        }
    # print(out)
    df = pd.DataFrame(out)
    print(df)


if __name__ == "__main__":
    main()
    # print(get_hyperparameter_grid()["small"].keys())
