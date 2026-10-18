class PaperUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # ============================================================
        # ENCODER
        # ============================================================

        # Input: 256 x 256 x 1

        self.enc1_conv1 = nn.Conv2d(
            1, 15, kernel_size=3, padding=1
        )

        self.enc1_conv2 = nn.Conv2d(
            15, 15, kernel_size=3, padding=1
        )

        self.pool1 = nn.MaxPool2d(2)

        # 128 x 128 x 15


        self.enc2_conv1 = nn.Conv2d(
            15, 30, kernel_size=3, padding=1
        )

        self.enc2_conv2 = nn.Conv2d(
            30, 30, kernel_size=3, padding=1
        )

        self.pool2 = nn.MaxPool2d(2)

        # 64 x 64 x 30


        self.enc3_conv1 = nn.Conv2d(
            30, 60, kernel_size=3, padding=1
        )

        self.enc3_conv2 = nn.Conv2d(
            60, 60, kernel_size=3, padding=1
        )

        self.pool3 = nn.MaxPool2d(2)

        # 32 x 32 x 60


        # ============================================================
        # BOTTLENECK
        # ============================================================

        self.bottleneck = nn.Conv2d(
            60, 60, kernel_size=3, padding=1
        )

        # ============================================================
        # DECODER
        # ============================================================

        # ------------------------------------------------------------
        # Decoder stage 1
        # 32 x 32 x 60 -> 32 x 32 x 60
        # -> 64 x 64 x 60
        # ------------------------------------------------------------

        self.dec1_conv = nn.Conv2d(
            60, 60, kernel_size=3, padding=1
        )

        self.up1 = nn.ConvTranspose2d(
            60, 60,
            kernel_size=2,
            stride=2
        )

        # ------------------------------------------------------------
        # Decoder stage 2
        # 64 x 64 x 60
        # -> 64 x 64 x 30
        # + encoder 64 x 64 x 30
        # -> 128 x 128 x 30
        # ------------------------------------------------------------

        self.dec2_conv = nn.Conv2d(
            60, 30, kernel_size=3, padding=1
        )

        self.up2 = nn.ConvTranspose2d(
            30, 30,
            kernel_size=2,
            stride=2
        )

        # ------------------------------------------------------------
        # Decoder stage 3
        # 128 x 128 x 30
        # -> 128 x 128 x 15
        # + encoder 128 x 128 x 15
        # -> 256 x 256 x 15
        # ------------------------------------------------------------

        self.dec3_conv = nn.Conv2d(
            30, 15, kernel_size=3, padding=1
        )

        self.up3 = nn.ConvTranspose2d(
            15, 15,
            kernel_size=2,
            stride=2
        )

        # ============================================================
        # OUTPUT
        # ============================================================

        self.output = nn.Conv2d(
            15, 1,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # ============================================================
        # ENCODER
        # ============================================================

        e1 = torch.relu(self.enc1_conv1(x))
        e1 = torch.relu(self.enc1_conv2(e1))

        # 256 x 256 x 15
        e1 = self.pool1(e1)

        # 128 x 128 x 15
        skip1 = e1


        e2 = torch.relu(self.enc2_conv1(e1))
        e2 = torch.relu(self.enc2_conv2(e2))

        # 128 x 128 x 30
        e2 = self.pool2(e2)

        # 64 x 64 x 30
        skip2 = e2


        e3 = torch.relu(self.enc3_conv1(e2))
        e3 = torch.relu(self.enc3_conv2(e3))

        # 64 x 64 x 60
        e3 = self.pool3(e3)

        # 32 x 32 x 60


        # ============================================================
        # BOTTLENECK
        # ============================================================

        x = torch.relu(self.bottleneck(e3))

        # 32 x 32 x 60


        # ============================================================
        # DECODER STAGE 1
        # ============================================================

        x = torch.relu(self.dec1_conv(x))

        # 32 x 32 x 60

        x = self.up1(x)

        # 64 x 64 x 60


        # ============================================================
        # DECODER STAGE 2
        # ============================================================

        x = torch.relu(self.dec2_conv(x))

        # 64 x 64 x 30

        x = x + skip2

        # 64 x 64 x 30

        x = self.up2(x)

        # 128 x 128 x 30


        # ============================================================
        # DECODER STAGE 3
        # ============================================================

        x = torch.relu(self.dec3_conv(x))

        # 128 x 128 x 15

        x = x + skip1

        # 128 x 128 x 15

        x = self.up3(x)

        # 256 x 256 x 15


        # ============================================================
        # OUTPUT
        # ============================================================

        x = self.output(x)

        # 256 x 256 x 1

        x = self.sigmoid(x)

        return x
