from functools import partial

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from simplepp import pipeline, StageKind, stage_scope


NUM_LAYERS = 4
BATCH_SIZE = 16
INPUT_SIZE = 4
HIDDEN_SIZE = 8
OUTPUT_SIZE = 4
LR = 0.01
TRAINSET_SIZE = 128 * BATCH_SIZE

NUM_STEPS = 100
LOG_EVERY = 10

NUM_ISLANDS = 4
NUM_MICROBATCH = 8
assert BATCH_SIZE % NUM_MICROBATCH == 0


def predict(params, xs):
  @jax.remat
  def layer(W, b, xs):
    return jax.nn.relu(xs @ W + b)

  for layer_idx, (W, b) in enumerate(params):
    # NB: Annotate model layers
    with stage_scope(StageKind.Model, layer_idx):
      xs = layer(W, b, xs)
  return xs


def mse(xs, ys):
  # NB: Annotate loss
  with stage_scope(StageKind.Loss):
    return jnp.mean(jnp.square(xs - ys))


def pipeline_step(mubatcher, params, xs, ys):
  mlp = lambda params, xs, ys: mse(predict(params, xs), ys)

  # NB: Define value-and-grad per microbatch
  def process_mubatch(mubatch):
    xs, ys = mubatch
    loss, grad = jax.value_and_grad(mlp)(params, xs, ys)
    return loss, grad
  # NB: Apply library-provided `mubatcher` which accumulates mubatch gradients
  loss, grad = mubatcher(process_mubatch)((xs, ys))

  def update_param(path, v, dv):
    # NB: Annotate parameter update
    with stage_scope(StageKind.Update, path[0].idx):
      return v - dv * LR
  params = jax.tree_util.tree_map_with_path(update_param, params, grad)

  return loss, params


def init_params(key):
  params = []
  sizes = (INPUT_SIZE,) + (HIDDEN_SIZE,) * NUM_LAYERS + (OUTPUT_SIZE,)
  for in_size, out_size in zip(sizes, sizes[1:]):
    key, key_W, key_b = jax.random.split(key, 3)
    params.append((
      jax.random.normal(key_W, (in_size, out_size)),
      jax.random.normal(key_W, (out_size,)),
    ))
  return params, key


def example_mlp():
  key = jax.random.PRNGKey(0)
  params, key = init_params(key)

  key, key_xs = jax.random.split(key)
  xs = jax.random.uniform(key_xs, (TRAINSET_SIZE, INPUT_SIZE))
  ys = jnp.square(xs)

  def next_batch():
    nonlocal key
    key, key_batch = jax.random.split(key)
    batch = jax.random.choice(key_batch, TRAINSET_SIZE, shape=(BATCH_SIZE,))
    # NB: Reshape batch to have a leading microbatch dimension
    return (
      xs[batch].reshape((NUM_MICROBATCH, -1, *xs.shape[1:])),
      ys[batch].reshape((NUM_MICROBATCH, -1, *ys.shape[1:])),
    )

  dummy_args = (params,) + next_batch()
  step_fn = pipeline(
    pipeline_step,
    dummy_args=dummy_args,
    num_mubatch=NUM_MICROBATCH,
    num_islands=NUM_ISLANDS,
  )
  step_fn = jax.jit(step_fn)

  for i in range(1, NUM_STEPS+1):
    xs_batch, ys_batch = next_batch()
    loss, params = step_fn(params, xs_batch, ys_batch)
    loss = loss.mean()
    if i % LOG_EVERY == 0:
      print(f'step={i:4d} loss={loss:.4}')
  
  assert loss <= 0.3


if __name__ == '__main__':
  example_mlp()
