import torch
from torch import nn


def run_accumulation_test(accumulation_steps=4, micro_batches=12):
    torch.manual_seed(0)
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    steps = 0
    optimizer.zero_grad()
    for i in range(micro_batches):
        x = torch.randn(8, 10)
        y = torch.randn(8, 1)
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        (loss / accumulation_steps).backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
    return steps


if __name__ == "__main__":
    s = run_accumulation_test(accumulation_steps=4, micro_batches=12)
    print(f"Optimization steps: {s} (expected 3)")
    assert s == 3, f"Expected 3 optimization steps, got {s}"
    print("Gradient accumulation test passed.")
