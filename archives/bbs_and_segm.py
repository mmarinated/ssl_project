

def bounding_boxes_to_segmentation(full_width, full_height, scale, bounding_boxes, categories):
    out = torch.zeros((full_height, full_width), dtype=torch.long)
    
    ind = np.indices((full_height, full_width))
    ind[0] = ind[0] - full_height / 2
    ind[1] = ind[1] - full_width / 2
    ind = np.moveaxis(ind, 0, 2)
    for i, b in enumerate(bounding_boxes.detach().data.numpy()):
        p = Path([b[:,0]*scale,b[:,1]*scale,b[:,3]*scale,b[:,2]*scale])
        g = p.contains_points(ind.reshape(full_width*full_height,2))
        g = np.flip(g.reshape(full_height, full_width), axis=1).T
        g = g.copy()
        out[g] = 1

    return out

def get_bounding_boxes_from_seg(segment_tensor, scale, full_height, full_width):
    contours, _ = cv2.findContours(to_np(segment_tensor).astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

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
    if len(filteredBoundRect) > 0:
        return torch.FloatTensor(filteredBoundRect).permute(0,2,1)
    else:
        return torch.zeros((1,2,4))