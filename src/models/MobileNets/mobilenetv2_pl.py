from torch import nn, optim
import pytorch_lightning as pl
from .mobilenetv2 import mobilenetv2

class MobileNetV2PL(pl.LightningModule):
    """ Pytorch Lightning wrapper for the EdgeNext Base network """
    def __init__(self, num_landmarks, batch_size=1, lr=0.001, weight_decay=0):
        super().__init__()
        self.model = mobilenetv2(num_landmarks=num_landmarks)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # print(batch.keys())
        inputs = batch['image'].float()
        targets = batch['landmarks'].float()
        outputs = self.model(inputs)#[0]
        loss = self.mse_loss(outputs, targets)
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch.keys())
        # "dict_keys(['image', 'sample_idx', 'imgpath', 'ids_ldm', 'bbox', 'bbox_raw', 'landmarks', 'visible', 'mask_ldm', 'landmarks_float', 'mask_ldm_float', 'img2map_scale', 'heatmap2D', 'cam_matrix', 'model3d', 'pose', 'model3d_proj'])"
        inputs = batch['image'].float()
        targets = batch['landmarks'].float()
        outputs = self.model(inputs)
        # print(f"Shapes: {outputs.shape} = {targets.shape}")
        loss = self.mse_loss(outputs, targets)
        self.log('val_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer