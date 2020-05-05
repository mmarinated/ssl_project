"""
get_info_by_idx : 
    to get predictions and targets for cars and roads:
    sample, predicted_cars_map, predicted_bbs, target_dict, predicted_road_map, road_image

plot_predicitons:
    plot 1 x 3 subplots: photos, target, predictions
"""

def get_info_by_idx(idx, dataset):
    with torch.no_grad():
        sample, target_dict, road_image = dataset[idx]
        sample = sample.cuda()
        
        predicted_cars_map, mu, var = model_loader.object_model(sample[None, :], is_training=False)
        predicted_cars_map = predicted_cars_map[0].cpu()

        predicted_bbs = model_loader.get_bounding_boxes(sample[None, :])[0].cpu()

        predicted_road_map = model_loader.get_binary_road_map(sample[None, :])[0].cpu()

    return sample, predicted_cars_map, predicted_bbs, target_dict, predicted_road_map, road_image

def plot_predicitons(sample, predicted_cars_map, predicted_bbs, target_dict, predicted_road_map, road_image):
    predicted_dict = {
        "bounding_box" : predicted_bbs,
        "category"     : torch.tensor([2] * len(predicted_bbs))
    }

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20,20))
    plot_utils.plot_photos(sample.cpu(), axis=ax0)

    ax1.set_title("Target")
    plot_utils.plot_bb(road_image, target_dict, axis=ax1)
    ax2.set_title("Prediction")
    plot_utils.plot_bb(predicted_road_map, predicted_dict, axis=ax2)
    ax2.imshow(predicted_cars_map, alpha=0.8)
    plt.show()