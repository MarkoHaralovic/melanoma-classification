import torch
from collections import defaultdict, deque
import logging
import torch.distributed as dist
import time
import datetime
from tensorboardX import SummaryWriter
from .distributed import is_dist_avail_and_initialized

class SmoothedValue(object):
   """Track a series of values and provide access to smoothed values over a
   window or the global series average.xxxxxxxxxxx
   """

   def __init__(self, window_size=20, fmt=None):
      if fmt is None:
         fmt = "{median:.4f} ({global_avg:.4f})"
      self.deque = deque(maxlen=window_size)
      self.total = 0.0
      self.count = 0
      self.fmt = fmt

   def update(self, value, n=1):
      self.deque.append(value)
      self.count += n
      self.total += value * n

   def synchronize_between_processes(self):
      """
      Warning: does not synchronize the deque!
      """
      if not is_dist_avail_and_initialized():
         return
      t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
      dist.barrier()
      dist.all_reduce(t)
      t = t.tolist()
      self.count = int(t[0])
      self.total = t[1]

   @property
   def median(self):
      d = torch.tensor(list(self.deque))
      return d.median().item()

   @property
   def avg(self):
      d = torch.tensor(list(self.deque), dtype=torch.float32)
      return d.mean().item()

   @property
   def global_avg(self):
      return self.total / self.count

   @property
   def max(self):
      return max(self.deque)

   @property
   def value(self):
      return self.deque[-1]

   def __str__(self):
      return self.fmt.format(
         median=self.median,
         avg=self.avg,
         global_avg=self.global_avg,
         max=self.max,
         value=self.value)


class MetricLogger(object):
   def __init__(self, delimiter="\t"):
      self.meters = defaultdict(SmoothedValue)
      self.delimiter = delimiter

   def update(self, **kwargs):
      for k, v in kwargs.items():
         if v is None:
               continue
         if isinstance(v, torch.Tensor):
               v = v.item()
         assert isinstance(v, (float, int))
         self.meters[k].update(v)

   def __getattr__(self, attr):
      if attr in self.meters:
         return self.meters[attr]
      if attr in self.__dict__:
         return self.__dict__[attr]
      raise AttributeError("'{}' object has no attribute '{}'".format(
         type(self).__name__, attr))

   def __str__(self):
      loss_str = []
      for name, meter in self.meters.items():
         loss_str.append(
               "{}: {}".format(name, str(meter))
         )
      return self.delimiter.join(loss_str)

   def synchronize_between_processes(self):
      for meter in self.meters.values():
         meter.synchronize_between_processes()

   def add_meter(self, name, meter):
      self.meters[name] = meter

   def log_every(self, iterable, print_freq, header=None):
      i = 0
      if not header:
         header = ''
      start_time = time.time()
      end = time.time()
      iter_time = SmoothedValue(fmt='{avg:.4f}')
      data_time = SmoothedValue(fmt='{avg:.4f}')
      space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
      log_msg = [
         header,
         '[{0' + space_fmt + '}/{1}]',
         'eta: {eta}',
         '{meters}',
         'time: {time}',
         'data: {data}'
      ]
      if torch.cuda.is_available():
         log_msg.append('max mem: {memory:.0f}')
      log_msg = self.delimiter.join(log_msg)
      MB = 1024.0 * 1024.0
      for obj in iterable:
         data_time.update(time.time() - end)
         yield obj
         iter_time.update(time.time() - end)
         if i % print_freq == 0 or i == len(iterable) - 1:
               eta_seconds = iter_time.global_avg * (len(iterable) - i)
               eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
               if torch.cuda.is_available():
                  logging.info(log_msg.format(
                     i, len(iterable), eta=eta_string,
                     meters=str(self),
                     time=str(iter_time), data=str(data_time),
                     memory=torch.cuda.max_memory_allocated() / MB))
               else:
                  logging.info(log_msg.format(
                     i, len(iterable), eta=eta_string,
                     meters=str(self),
                     time=str(iter_time), data=str(data_time)))
         i += 1
         end = time.time()
      total_time = time.time() - start_time
      total_time_str = str(datetime.timedelta(seconds=int(total_time)))
      logging.info('{} Total time: {} ({:.4f} s / it)'.format(
         header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
   def __init__(self, log_dir):
      self.writer = SummaryWriter(logdir=log_dir)
      self.step = 0

   def set_step(self, step=None):
      if step is not None:
         self.step = step
      else:
         self.step += 1

   def update(self, head='scalar', step=None, **kwargs):
      for k, v in kwargs.items():
         if v is None:
               continue
         if isinstance(v, torch.Tensor):
               v = v.item()
         assert isinstance(v, (float, int))
         self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

   def flush(self):
      self.writer.flush()


class WandbLogger(object):
   def __init__(self, args):
      self.args = args

      try:
         import wandb
         self._wandb = wandb
      except ImportError:
         raise ImportError(
               "To use the Weights and Biases Logger please install wandb."
               "Run `pip install wandb` to install it."
         )

      # Initialize a W&B run 
      if self._wandb.run is None:
         self._wandb.init(
               project=args.project,
               config=args
         )

   def log_epoch_metrics(self, metrics, commit=True):
      """
      Log train/test metrics onto W&B.
      """
      # Log number of model parameters as W&B summary
      self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
      metrics.pop('n_parameters', None)

      # Log current epoch
      self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
      metrics.pop('epoch')

      for k, v in metrics.items():
         if 'train' in k:
               self._wandb.log({f'Global Train/{k}': v}, commit=False)
         elif 'test' in k:
               self._wandb.log({f'Global Test/{k}': v}, commit=False)

      self._wandb.log({})

   def log_checkpoints(self):
      output_dir = self.args.output_dir
      model_artifact = self._wandb.Artifact(
         self._wandb.run.id + "_model", type="model"
      )

      model_artifact.add_dir(output_dir)
      self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

   def set_steps(self):
      # Set global training step
      self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
      # Set epoch-wise step
      self._wandb.define_metric('Global Train/*', step_metric='epoch')
      self._wandb.define_metric('Global Test/*', step_metric='epoch')
