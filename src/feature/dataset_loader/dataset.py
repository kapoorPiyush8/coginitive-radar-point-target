class RadarRDMDataset(Dataset):

    def __init__(self, data_dir, num_scenes=1000):

        self.data_dir = Path(data_dir)
        self.num_scenes = num_scenes
        self.batch_size = 100

        self.rdm_batches = []
        self.label_batches = []

        for start in range(0, num_scenes, self.batch_size):

            end = start + self.batch_size - 1

            rdm_path = (
                self.data_dir /
                f"rdm_{start:05d}_{end:05d}.npy"
            )

            label_path = (
                self.data_dir /
                f"labels_{start:05d}_{end:05d}.npy"
            )

            self.rdm_batches.append(
                np.load(rdm_path, mmap_mode="r")
            )

            self.label_batches.append(
                np.load(label_path, mmap_mode="r")
            )

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):

        batch_idx = idx // self.batch_size
        local_idx = idx % self.batch_size

        # Load raw RDM
        rdm = np.array(
            self.rdm_batches[batch_idx][local_idx],
            dtype=np.float32,
            copy=True
        )

        label = np.array(
            self.label_batches[batch_idx][local_idx],
            dtype=np.float32,
            copy=True
        )

        # Convert RDM to tensor
        rdm_tensor = torch.from_numpy(rdm).unsqueeze(0)

        # Paper-specified spatial resizing
        rdm_tensor = F.interpolate(
            rdm_tensor.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        
       

        label_tensor = torch.from_numpy(label).unsqueeze(0)

        return rdm_tensor, label_tensor
