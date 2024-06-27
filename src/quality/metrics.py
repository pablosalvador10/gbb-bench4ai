from typing import Any

def accuracy(y_pred: Any, y_true: Any) -> int:
    if y_pred == y_true:
        return 1
    else:
        return 0