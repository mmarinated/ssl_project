"""
functions from this file needed to be run once to generate 
preprocess images / labels in data/ folder
"""


def create_label_data(scene_id):
    dist_WW = create_dist_WW()  

    
    for sample_id in tqdm(range(NUM_SAMPLE_PER_SCENE)):
        PATH_TMP = f"{PATH_TO_DATA}/scene_{scene_id}/sample_{sample_id}"
        top_down_segm_WW = cv2.imread(f"{PATH_TMP}/top_down_segm.png", cv2.IMREAD_UNCHANGED)
        height_WW = create_height_WW(top_down_segm_WW)
        
        for idx, camera_name in enumerate(CAM_NAMES):
            # photo_hw = cv2.imread(f"../data/scene_{scene_id}/sample_{sample_id}/{name}.jpeg", cv2.IMREAD_UNCHANGED)
            segm_hw = create_segm_hw(top_down_segm_WW, height_WW, idx)
            segm_hw[segm_hw == NO_IMAGE_LABEL] = max(top_down_segmentation.VALUE_TO_CATEGORY.keys()) + 1
            dist_hw = create_segm_hw(dist_WW,          height_WW, idx)

            cv2.imwrite(f"{PATH_TMP}/SEGM_{camera_name}.png", segm_hw)
            cv2.imwrite(f"{PATH_TMP}/DIST_{camera_name}.png", dist_hw)



def create_label_data_road_only(scene_id):
    def get_road_image(PATH_TMP):
        ego_image = cv2.imread(f"{PATH_TMP}/ego.png", cv2.IMREAD_UNCHANGED)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = to_np(convert_map_to_road_map(ego_image)).astype(int)
        return road_image

    height_WW = np.zeros((EGO_IMAGE_SIZE, EGO_IMAGE_SIZE))
    for sample_id in tqdm(range(NUM_SAMPLE_PER_SCENE)):
        PATH_TMP = f"{PATH_TO_DATA}/scene_{scene_id}/sample_{sample_id}"
        top_down_segm_WW = get_road_image(PATH_TMP) # TODO cv2.imread(f"{PATH_TMP}/top_down_segm.png", cv2.IMREAD_UNCHANGED)
        # return top_down_segm_WW
        
        for idx, camera_name in enumerate(CAM_NAMES):
            segm_hw = create_segm_hw(top_down_segm_WW, height_WW, idx)
            # 1 is road 0 is not road
            segm_hw[segm_hw != 1] = 0 # TODO max(top_down_segmentation.VALUE_TO_CATEGORY.keys()) + 1
            cv2.imwrite(f"{PATH_TMP}/ROAD_SEGM_{camera_name}.png", segm_hw)
            

def create_data_all(n_jobs=8, slc=slice(None), debug=False):    
    if debug:
        for idx in tqdm(LABELED_SCENE_INDEX[slc]):
            create_label_data(idx)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(create_label_data)(scene_id) 
            for scene_id in LABELED_SCENE_INDEX[slc]
        )
