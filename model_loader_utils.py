import cv2
import torch
import numpy as np

def get_bounding_boxes_from_seg(segment_tensor, scale, full_height, full_width):
    _, contours, _ = cv2.findContours(to_np(segment_tensor).astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        
    def convert(value, dim):
        if dim == "x":
            out = ( value - full_width / 2 ) / scale
        elif dim == "y":
             out = -( value - full_height / 2 ) / scale
        return out
    for i in range(len(boundRect)):
        boundRect[i] = [[convert(boundRect[i][0] + boundRect[i][2], "x"), convert(boundRect[i][1], "y")], 
                        [convert(boundRect[i][0] + boundRect[i][2], "x"), convert(boundRect[i][1] +boundRect[i][3], "y")],
                        [convert(boundRect[i][0], "x"), convert(boundRect[i][1], "y")],
                        [convert(boundRect[i][0], "x"), convert(boundRect[i][1] + boundRect[i][3], "y")]
                        ]
    filteredBoundRect = []
    for i in range(len(boundRect)):
        if ( ( boundRect[i][0][0] - boundRect[i][2][0] ) > 1) and (( boundRect[i][0][1] - boundRect[i][1][1] )>1):
            filteredBoundRect.append(boundRect[i])
    del boundRect
    del contours_poly
    del _
    if len(filteredBoundRect) > 0:
        return torch.FloatTensor(filteredBoundRect).permute(0,2,1)
    else:
        return torch.zeros((1,2,4))
    
def to_np(x):
    return x.detach().cpu().data.squeeze().numpy()