# Gradient Accumulation in F2LLM

## How Gradient Accumulation Works in This Codebase

1. Set `gradient_accumulation_steps` in the config.json and arguments.py file (default is 1, meaning no accumulation)
   - e.g: `"gradient_accumulation_steps": 4` will accumulate gradients over 4 micro-batches


2. `utils.py`:
   ```python
   # Scale loss by gradient accumulation steps to maintain same effective learning rate
   loss_total = loss_total / args.gradient_accumulation_steps

   # Update step only after gradient_accumulation_steps
   if (completed_steps + 1) % args.gradient_accumulation_steps == 0:
       optimizer.step()
       lr_scheduler.step()
       optimizer.zero_grad()
   ```
   - Without accumulation: Process 1 batch of size N → compute loss → update parameters
   - With accumulation: Process 4 micro-batches of size N/4 → accumulate gradients → update parameters

   Both result in same parameter update if learning rate is properly scaled


## Example

Let's say you have:
- Desired effective batch size: 32
- GPU memory only allows: 8 samples per batch

**Without Gradient Accumulation**:
- You're limited to batch size 8
- Effective batch size = 8
- May result in suboptimal training dynamics

**With Gradient Accumulation (steps=4)**:
- Process 4 micro-batches of size 8 each
- Effective batch size = 32 (4 × 8)
- Same training dynamics as a batch size of 32
- Better gradient estimates due to larger effective batch size

## Configuration Example

To use gradient accumulation, modify your config file:
```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 4,
  // This gives you an effective batch size of 32 (8 * 4)
  // while only using memory for 8 samples at a time
}
```