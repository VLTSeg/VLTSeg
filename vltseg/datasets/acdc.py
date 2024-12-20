from mmseg.registry import DATASETS
from mmseg.datasets import CityscapesDataset


@DATASETS.register_module()
class ACDCDataset(CityscapesDataset):
    """ACDCDataset dataset."""

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
