class PaperDiceLoss(nn.Module):
    """
    Dice loss exactly following the equation given in the paper.

        D = 2 * sum(p*y) /
            [sum(p^2) + sum(y^2)]

        Loss = 1 - D

    Dice is calculated independently for each image
    and then averaged over the batch.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):

        # predictions: [B, 1, H, W]
        # targets:     [B, 1, H, W]

        predictions = predictions.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        numerator = 2.0 * torch.sum(
            predictions * targets,
            dim=1
        )

        denominator = (
            torch.sum(predictions ** 2, dim=1)
            +
            torch.sum(targets ** 2, dim=1)
        )

        dice = numerator / (denominator + 1e-8) 
        # add 1e-8 to avoid division by zero in dice calculation

        loss = 1.0 - dice

        return loss.mean()
