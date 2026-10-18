dataset = RadarRDMDataset(
    dataset_dir,
    num_scenes=1000
)

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [800, 200],
    generator=generator
)
