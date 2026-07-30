import torch


class EarlyStopper(object):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

    :param patience: Number of events to wait if no improvement and then stop the training.
    :type patience: int
    :param min_delta: A minimum decrease in the score to qualify as an improvement. Default is 0.0.
    :type min_delta: float, optional
    :param cumulative_delta: If True, `min_delta` defines a decrease since the last `patience` reset. Default is False.
    :type cumulative_delta: bool, optional
    :param checkpoint_path: File used for the best-model checkpoint. Defaults to
        ``"checkpoint.pth"`` for backward compatibility.
    :type checkpoint_path: str, optional
    :return: An instance of EarlyStopper.
    :rtype: EarlyStopper

    .. note::
       The score checked by :meth:`check_loss` is a **loss** (lower is better): a model checkpoint
       is saved whenever the score improves on the best one by more than ``min_delta``, and
       training stops after ``patience`` consecutive non-improving epochs.

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
        checkpoint_path: str = "checkpoint.pth",
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
        self.checkpoint_path = checkpoint_path

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
        elif score >= self.best_score - self.min_delta:
            # No improvement of at least min_delta over the best checkpointed score. The best
            # score is deliberately NOT updated here: it tracks the saved checkpoint, so the bar
            # can never slide toward worse models.
            self.counter += 1
            print("Early stopping counter %d of %d" % (self.counter, self.patience))
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
        torch.save(model.state_dict(), self.checkpoint_path)
        """_summary_
        """
