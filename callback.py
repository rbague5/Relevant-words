from gensim.models.callbacks import CallbackAny2Vec


class MonitorLoss(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        current_loss = round(loss - self.loss_previous_step, 4)
        self.loss_previous_step = loss
        print(f"Epoch {self.epoch}, loss: {current_loss}")
        self.epoch += 1
