class SegmentationDataset(AbstractLabeledDataset):
    def __init__(self, ... input_image_name):
        """
        input_image_name = kornia_preprossed_BEV.png
        """
        super(....)

        self.input_fname = input_image_name
        self.output_fname = output_image_name
    
    
    def __getitem__(self, index):
        """
        return 
            image_tensor, target, road_image, (extra or None)
        """

        scene_id, sample_id, sample_path = self._get_ids_and_path(index)
        
        out = {}

        out["input_3WW"] = load_image(scene_id,  sample_id, self.input_fname)        
        out["output_WW"] = load_image(scene_id, sample_id, self.output_fname)

        if self.load_original_images: 
            out["image_63hw"] = self._get_images(sample_path)
        if self.load_bounding_boxes:
            out["target"], road_image, ego_image, data_entries =\
                self._get_target_road_ego_image(scene_id, sample_id, sample_path)

        return out