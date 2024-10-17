# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import CityscapesDataset


@DATASETS.register_module()
class SynthiaDataset(CityscapesDataset):
    """SynthiaDataset dataset."""

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)