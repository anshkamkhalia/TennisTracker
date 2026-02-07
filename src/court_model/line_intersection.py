def line_intersection(line1, line2):
    """computes the intersection of 2 lines"""
    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # compute determinant
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None # lines are parallel
    
    # intersection formulas
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return [int(px), int(py)]
