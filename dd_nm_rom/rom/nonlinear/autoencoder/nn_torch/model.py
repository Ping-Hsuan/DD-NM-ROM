import os
import torch

from . import optimization as optim
from dd_nm_rom import backend as bkd


class Model(object):

  def __init__(
    self,
    net=None,
    data=None,
    path="./"
  ):
    self.net = net
    self.data = data
    # Attributes for optimization
    self.loss = None
    self.optimizer = None
    self.callbacks = None
    # Training state
    logs_ids = ["loss"]
    if (self.data.valid is not None):
      logs_ids.append("val_loss")
    self.train_state = optim.TrainState()
    self.train_state.init_logs(logs_ids)
    # Define paths
    self.path = path
    self.dirs = {
      "save": self.path + "/saving/",
      "train": self.path + "/training/",
      "ckpt": self.path + "/training/ckpt/"
    }
    for d in self.dirs.values():
      os.makedirs(d, exist_ok=True)
    # Control vars
    self.is_compiled = False
    self.stop_training = False

  # Compiling
  # ---------------------------------
  def compile(
    self,
    optimizer="adam",
    lr=1e-3,
    decay=None,
    loss="mse",
    monitor="loss",
    reduction="sum",
    callbacks=None
  ):
    print("Compiling the model ...")
    # Write nn summary
    self.net.summary(filename=self.dirs["save"] + "/summary.txt")
    # Initializing loss function
    self.loss = optim.losses.get(loss, reduction=reduction)
    # Monitor metrics
    self.monitor = monitor
    options = list(self.train_state.logs.keys())
    if (self.monitor not in options):
      raise ValueError(f"Monitor metrics not valid. Please choose: {options}")
    # Initializing the optimizer
    self.optimizer, self.lr_scheduler = optim.optimizers.get(
      self.net.parameters(), optimizer, lr=lr, decay=decay
    )
    # Callbacks
    self.callbacks = optim.callbacks.get_callbacks(callbacks)
    self.callbacks.set_model(self)
    self.is_compiled = True

  # Training
  # ---------------------------------
  def train(
    self,
    epochs=100,
    display_freq=1,
    saving=True
  ):
    if self.net.trainable:
      print("Training the model ...")
      # Training state
      self.train_state.epochs = epochs
      self.train_state.display_freq = display_freq
      # Training
      self.callbacks.set_display_freq(display_freq)
      self.callbacks.on_train_begin()
      self.train_sgd()
      self.callbacks.on_train_end()
    else:
      print("Warning! Training skipped since the model is not trainable!")
    # Saving
    if saving:
      print("Saving the model ...")
      self.save()

  def train_sgd(self):
    for _ in range(self.train_state.epochs):
      # On epoch begin calls
      self.train_state.on_epoch_begin()
      self.callbacks.on_epoch_begin()
      self.data.on_epoch_begin()
      # Train step
      self.net.train(mode=True)
      for batch in self.data.batches:
        # On batch begin calls
        self.callbacks.on_batch_begin()
        # Training step
        self.train_step(batch.to(bkd.device()))
        # On batch end calls
        self.callbacks.on_batch_end()
      # Test step
      with torch.set_grad_enabled(False):
        self.evaluate()
      # Update lr
      self.update_lr()
      # On epoch end calls
      self.train_state.on_epoch_end()
      self.callbacks.on_epoch_end()
      self.train_state.epoch += 1
      if self.stop_training:
        break

  def train_step(self, data):
    # Run forward pass
    loss = self.evaluate_step(data)
    # Run backwards pass
    self.optimizer.zero_grad()
    loss.backward()
    # Update parameters
    self.optimizer.step()
    # Update logs
    self.train_state.on_batch_end({"loss": loss})

  def update_lr(self):
    if (self.lr_scheduler is not None):
      if self.lr_scheduler.metrics_needed:
        metrics = self.train_state.logs[self.monitor]
        self.lr_scheduler.step(metrics)
      else:
        self.lr_scheduler.step()

  # Testing
  # ---------------------------------
  def evaluate(self):
    if (self.data.valid is not None):
      self.net.train(mode=False)
      for batch in self.data.batches_valid:
        loss = self.evaluate_step(batch.to(bkd.device()))
        # Update logs
        self.train_state.on_batch_end({"val_loss": loss})

  def evaluate_step(self, data):
    return self.loss(self.net(data), data)

  # Saving
  # ---------------------------------
  def save(self, filename=None):
    if (filename is None):
      filename = self.dirs["save"] + "/model_last"
    torch.save(self.net.state_dict(), filename+"_torch.p")
    torch.save(self.net.state_dict_np(), filename+"_numpy.p")
