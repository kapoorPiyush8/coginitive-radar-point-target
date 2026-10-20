from torch.utils.data import random_split

# Fixed, reproducible split
generator = torch.Generator().manual_seed(42)

train_size = 16_000
val_size = 4_000

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

#for validaion check only

print("Training scenes  :", len(train_dataset))
print("Validation scenes:", len(val_dataset))
