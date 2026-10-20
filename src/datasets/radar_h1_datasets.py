class RadarH1Dataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        self.rdm_files = sorted(
            self.data_dir.glob("rdm_*.npy")
        )

        self.label_files = sorted(
            self.data_dir.glob("labels_*.npy")
        )

        assert len(self.rdm_files) == len(self.label_files), \
            "RDM/label file count mismatch"

        # Each file contains 50 scenes
        self.scenes_per_file = 50

        self.num_scenes = (
            len(self.rdm_files) *
            self.scenes_per_file
        )

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):

        file_idx = idx // self.scenes_per_file
        scene_idx = idx % self.scenes_per_file

        rdm = np.load(
            self.rdm_files[file_idx]
        )[scene_idx]

        label = np.load(
            self.label_files[file_idx]
        )[scene_idx]

        # -----------------------------
        # RDM preprocessing
        # -----------------------------

        x = torch.from_numpy(rdm).float()

        x = torch.clamp(
            x,
            min=-40.0,
            max=0.0
        )

        x = (x + 40.0) / 40.0

        # 69 x 1334 -> 256 x 256
        x = F.interpolate(
            x.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # -----------------------------
        # Label
        # -----------------------------

        y = torch.from_numpy(label).float()
        y = y.unsqueeze(0)

        return x, y
