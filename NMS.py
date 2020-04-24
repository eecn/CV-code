import numpy as np

def nms(dets, thresh):
    '''
    Reference: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    Non-maximum suppression.
    Parameters:
    ----------
    dets : A vector of size n × 5
        The 1st column is the x1 coordinates of all rectangles,
        The 2nd column is the y1 coordinates of all rectangles,
        The 3rd column is the x2 coordinates of all rectangles,
        The 4th column is the y2 coordinates of all rectangles,
        The last column is the confidence scores.
    thresh : Threshold : float
    Returns:
    -------
    list
        List of rectangular ids after non-maximum suppression.  
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
if __name__ == '__main__':
    dets = np.array([[100,120,170,200,0.98],
                     [20,40,80,90,0.99],
                     [20,38,82,88,0.96],
                     [200,380,282,488,0.9],
                     [19,38,75,91, 0.8]])
    thresh = 0.5

    res = nms(dets, thresh)
    print(res)
