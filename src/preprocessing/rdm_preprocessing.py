# ============================================================
# 8 october
# ============================================================

# Paper-style input preprocessing
x = torch.clamp(x, min=-40.0, max=0.0)
x = (x + 40.0) / 40.0

# Resize RDM: 69x1334 -> 256x256
x = F.interpolate(
    x.unsqueeze(1),
    size=(256, 256),
    mode="bilinear",
    align_corners=False
)

# Add channel dimension to labels
y = y.unsqueeze(1)
