def bb_intersection_over_union(boxA, boxB):
    '''
    Reference: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters:
    ----------
    boxA : list or array,the element is int
        [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    boxB : list or array,the element is int
        [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns:
    -------
    float
        in [0, 1]  
    '''
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # not intersect
    if interArea == 0:
        return 0
    
	# compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou
if __name__ == '__main__':
    # Pointing out a wrong IoU implementation in https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    boxA = [1, 1, 102., 102.]
    boxB = [10, 10, 120, 120]

    print('the IOU between boxA and boxB is ',bb_intersection_over_union(boxA, boxB))
