from .ConvNeXt.convnext_pl import ConvNeXtPL
from .SHG.StackedHourglass_pl import LitSHGPL
from .MobileViTs.edgenext_l_pl import EdgeNextPL
from .MobileNets.mobilenetv2_pl import MobileNetV2PL 

def get_model(name:str, num_landmarks = 98, **hparams):
    name = name.lower()

    if name.startswith('shg'):
        model = LitSHGPL(num_modules=1, num_landmarks=num_landmarks, **hparams)

    elif name.startswith('edgenext'):
        size = name.partition("edgenext_")[2]
        model = EdgeNextPL(num_landmarks, size=size, **hparams)

    elif name.startswith('mobilenetv2'):
        model = MobileNetV2PL(num_landmarks, **hparams)

    elif name.startswith('convnext'):
        model = ConvNeXtPL(num_landmarks, **hparams)

    else:
        raise NotImplementedError

    return model
