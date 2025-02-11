from .criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss
from .parts.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from .multimae import MultiMAE, MultiViT
from .parts.output_adapters import (ConvNeXtAdapter, DPTOutputAdapter,
                              LinearOutputAdapter,
                              SegmenterMaskTransformerAdapter,
                              SpatialOutputAdapter)
