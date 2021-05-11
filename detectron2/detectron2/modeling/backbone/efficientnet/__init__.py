__version__ = "0.6.1"
from .efficientnet import EfficientNet
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
    MemoryEfficientSwish,
    Swish
)

from .utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
