from .SHG.StackedHourglass_pl import LitSHG
from .MobileViTs.edgenext_l_pl import EdgeNextPL
from .MobileNets.mobilenetv2_pl import MobileNetV2PL 

def get_model(name, **hparams):
    # Stacked Hourglass: SHG_<num_modules>_<num_landmarks>
    if name.startswith('SHG'):
        num_modules = int(name.split('_')[1])
        num_landmarks = int(name.split('_')[2])
        model = LitSHG(num_modules=num_modules, num_landmarks=num_landmarks, **hparams)
    elif name.startswith('EdgeNext'):
        size = str(name.split('_')[1])
        num_landmarks = int(name.split('_')[2])
        model = EdgeNextPL(num_landmarks, size=size, **hparams)
    elif name.startswith('MobileNetV2'):
        num_landmarks = int(name.split('_')[1])
        model = MobileNetV2PL(num_landmarks, **hparams)
    else:
        raise NotImplementedError

    return model
