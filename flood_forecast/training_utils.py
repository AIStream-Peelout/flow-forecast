import torch


class EarlyStopper(object):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

    :param patience: Number of events to wait if no improvement and then stop the training.
    :type patience: int
    :param min_delta: A minimum decrease in the score to qualify as an improvement. Default is 0.0.
    :type min_delta: float, optional
    :param cumulative_delta: If True, `min_delta` defines a decrease since the last `patience` reset. Default is False.
    :type cumulative_delta: bool, optional
    :return: An instance of EarlyStopper.
    :rtype: EarlyStopper

    .. note::
       The original docstring implies an "improvement" is a higher score.
       However, the logic in `check_loss` (specifically `score + self.min_delta >= self.best_score`) suggests the
       score being checked is a **loss** or a metric where **lower is better** (since it saves the model when `score < self.best_score`).
       I've updated the description to reflect the *actual* logic of the provided code where it's used to check for a decrease in loss.

    Examples:
    .. code-block:: python

        from ignite.engine import Engine, Events
        from ignite.handlers import EarlyStopping
        def score_function(engine):
            val_loss = engine.state.metrics['nll']
            return -val_loss
        handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        evaluator.add_event_handler(Events.COMPLETED, handler)
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def check_loss(self, model, validation_loss) -> bool:
        """Checks the validation loss against the best recorded score to determine if training should stop.

        :param model: The PyTorch model instance whose state should be saved.
        :type model: torch.nn.Module
        :param validation_loss: The current validation loss (score). Lower is better.
        :type validation_loss: float
        :return: True if training should continue, False if training should stop.
        :rtype: bool
        """
        score = validation_loss
        if self.best_score is None:
            self.save_model_checkpoint(model)
            self.best_score = score

        elif score + self.min_delta >= self.best_score:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            print(self.counter)
            if self.counter >= self.patience:
                return False
        else:
            self.save_model_checkpoint(model)
            self.best_score = score
            self.counter = 0
        return True

    def save_model_checkpoint(self, model):
        """Saves the current model's state dictionary to a checkpoint file.

        :param model: The PyTorch model instance to save.
        :type model: torch.nn.Module
        :return: None
        :rtype: None
        """
        torch.save(model.state_dict(), "checkpoint.pth")
        """_summary_
        """