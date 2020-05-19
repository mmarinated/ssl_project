import torch

from shapely.geometry import Polygon


def compute_ats_bounding_boxes(pred_boxes_p24, targ_boxes_t24):
    n_p, n_t  = pred_boxes_p24.size(0), targ_boxes_t24.size(0)

    def get_min_max_x_y_n(boxes_n24):
        min_x_n, max_x_n = boxes_n24[:, 0].min(dim=1)[0], boxes_n24[:, 0].max(dim=1)[0]
        min_y_n, max_y_n = boxes_n24[:, 1].min(dim=1)[0], boxes_n24[:, 1].max(dim=1)[0]
        return min_x_n, max_x_n, min_y_n, max_y_n

    min_x_p, max_x_p, min_y_p, max_y_p = get_min_max_x_y_n(pred_boxes_p24)
    min_x_t, max_x_t, min_y_t, max_y_t = get_min_max_x_y_n(targ_boxes_t24)

    condition_matrix_pt = (
             (max_x_p[:, None] > min_x_t[None, :])
        and  (min_x_p[:, None] < max_x_t[None, :])
        and  (max_y_p[:, None] > min_y_t[None, :])
        and  (min_y_p[:, None] < max_y_t[None, :])
    )

    iou_matrix_pt = torch.zeros(n_p, n_t)
    for p_idx in range(n_p):
        for t_idx in range(n_t):
            if condition_matrix_pt[p_idx, t_idx]:
                iou_matrix_pt[p_idx][t_idx] = compute_iou(pred_boxes_p24[p_idx], targ_boxes_t24[t_idx])

    iou_max_t = iou_matrix_pt.max(dim=0)[0]


    thresholds_k = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    weight_k     = 1. / thresholds_k
    tp_kt = (iou_max_t[None, :] > thresholds_k[:, None]).float()
    tp_k  = tp_kt.sum(dim=1)
    threat_score_k = tp_k * 1.0 / (n_p + n_t - tp_k)
    
    return weight_k.dot(threat_score_k) / weight_k.sum()

def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()

    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)

def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull
    
    return a.intersection(b).area / a.union(b).area
