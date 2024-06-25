
def accuracy(y_pred, y_true) -> int:
    if y_pred == y_true:
        return 1
    else:
        return 0