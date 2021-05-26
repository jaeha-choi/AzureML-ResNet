class EarlyStopping():
    def __init__(self, patience=5, min_delta=0.01):
        self._patience=patience
        self._min_delta=min_delta

        self._loss_best=None
        self._count=0

    def __call__(self, val_loss):
        if self._loss_best is None:
            self._loss_best = val_loss
        elif self._loss_best - val_loss > self._min_delta: # improvement
            self._loss_best = val_loss
        elif self._loss_best - val_loss < self._min_delta: # worsen
            self._count += 1
            print(f"[Early Stopping] No improvements {self._count}/{self._patience}. Val loss: {val_loss}")

            if self._count == self._patience:
                print("[Early Stopping] Patience limit reached.")
                return False
        
        return True