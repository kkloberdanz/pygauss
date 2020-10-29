def ordinary_least_squares(x, y):
    mean_x = x.mean()
    mean_y = y.mean()
    m = (((x - mean_x) * (y - mean_y)).sum() /
         ((x - mean_x).square()).sum())
    b = mean_y - m * mean_x
    return m, b
