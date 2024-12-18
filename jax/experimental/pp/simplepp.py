from collections import defaultdict, deque
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache, partial, wraps

import jax
from jax._src.interpreters import partial_eval as pe
import jax._src.linear_util as lu
from jax._src.util import safe_zip
from jax.api_util import flatten_fun, flatten_fun_nokwargs
from jax.core import Atom, ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var
from jax.tree_util import tree_flatten, tree_leaves, tree_leaves_with_path, tree_map, tree_structure, tree_unflatten

from typing import Any, Callable, ClassVar, Iterable, Optional, Sequence, TypeVar

from jaxpr_custom_print import Colors, annotate_eqns, jaxpr_custom_print
import simplempmd

T = TypeVar('T')


### mubatch_scope helpers

_mubatch_scope_prefix = '__mubatch-'

def mubatch_scope_name(index: int) -> str:
  return f'{_mubatch_scope_prefix}{index}'

@cache
def unapply_mubatch_scope_name(scope_name: str) -> Optional[int]:
  if not scope_name.startswith(_mubatch_scope_prefix):
    return None
  name = scope_name[len(_mubatch_scope_prefix):]
  return int(name)


### stage_scope helpers

_stage_scope_prefix = '__stage-'
StageKind = Enum('StageKind', 'Model Loss Update')

def stage_scope_name(kind: StageKind, index: Optional[int]) -> str:
  if kind == StageKind.Loss:
    name = kind.name
  else:
    assert index is not None
    name = f'{kind.name}{index}'
  return f'{_stage_scope_prefix}{name}'

@cache
def unapply_stage_scope_name(
    scope_name: str
) -> Optional[tuple[StageKind, Optional[int]]]:
  if not scope_name.startswith(_stage_scope_prefix):
    return None
  name = scope_name[len(_stage_scope_prefix):]
  if name.startswith(StageKind.Model.name):
    return StageKind.Model, int(name[len(StageKind.Model.name):])
  elif name.startswith(StageKind.Update.name):
    return StageKind.Update, int(name[len(StageKind.Update.name):])
  elif name == StageKind.Loss.name:
    return StageKind.Loss, None
  else:
    assert False


### Regions

@dataclass(frozen=True)
class Stage:
  kind: StageKind
  index: Optional[int]
  fwd: bool

  def __post_init__(self):
    assert self.kind != StageKind.Update or self.fwd


@dataclass(frozen=True)
class Region:
  """Regions are inferred from annotations and map to MPMD tasks."""
  mubatch_idx: Optional[int]
  stage: Stage

  def __post_init__(self):
    assert (self.mubatch_idx is None) == (self.stage.kind == StageKind.Update)

  @property
  def label(self):
    mubatch_idx = self.mubatch_idx
    kind, index, fwd = self.stage.kind, self.stage.index, self.stage.fwd
    mubatch_prefix = f'm{mubatch_idx}/' if mubatch_idx is not None else ''
    index_str = str(index) if index is not None else ''
    dir_str = 'F' if fwd else 'B'
    return f'{mubatch_prefix}{kind.name}{index_str}{dir_str}'

  def __repr__(self):
    return self.label


MaybeRegion = Optional[Region]


### Region inference

def var_defn_and_uses(jaxpr):
  var_defn: dict[Var, JaxprEqn] = {}
  var_uses: dict[Var, set[JaxprEqn]] = defaultdict(set)
  for eqn in jaxpr.eqns:
    for invar in vars_only(eqn.invars):
      var_uses[invar].add(eqn)
    for outvar in eqn.outvars:
      var_defn[outvar] = eqn
  return var_defn, var_uses


def get_annotated_region(scopes: list[str], fwd, eqn) -> MaybeRegion:
  mubatch_scope = None
  stage_scope = None
  for scope in scopes:
    if (unapplied := unapply_mubatch_scope_name(scope)) is not None:
      if mubatch_scope is not None and unapplied != mubatch_scope:
        raise ValueError(
          f'found eqn {eqn} @ {eqn.source_info.traceback} nested in multiple ' \
          f'mubatch scopes: {unapplied} and {mubatch_scope} ({scopes=})'
        )
      mubatch_scope = unapplied
    elif (unapplied := unapply_stage_scope_name(scope)) is not None:
      if stage_scope is not None:
        raise ValueError(
          f'found eqn {eqn} @ {eqn.source_info.traceback} nested in multiple ' \
          f'stage scopes: {unapplied} and {stage_scope} ({scopes=})'
        )
      stage_scope = unapplied

  if stage_scope:
    kind, index = stage_scope
    mubatch_idx: Optional[int] = mubatch_scope
    if mubatch_idx is None and kind in (StageKind.Model, StageKind.Loss):
      raise ValueError(
        f'found eqn {eqn} @ {eqn.source_info.traceback} with invalid stage ' \
        f'kind {kind} that must be nested in mubatch scope, but is not.'
      )
    return Region(mubatch_idx, Stage(kind, index, fwd))
  else:
    return None


def infer_regions(
    closed_jaxpr: ClosedJaxpr,
) -> tuple[list[Region], dict[JaxprEqn, MaybeRegion]]:
  main_jaxpr = closed_jaxpr.jaxpr

  # Infer regions for equations
  regions: dict[Region, tuple[()]] = {}
  eqn_region: dict[JaxprEqn, MaybeRegion] = defaultdict(lambda: None)

  # Assign regions purely based on mubatch_scope and stage_scope in name stack
  for eqn in main_jaxpr.eqns:
    scopes = [scope.name for scope in eqn.source_info.name_stack.stack]
    num_transpose = sum(1 for scope in scopes if scope == 'transpose')
    if num_transpose > 1:
      raise ValueError(f'higher-degree transpose unsupported, got {scopes=}')
    fwd = num_transpose == 0
    if (region := get_annotated_region(scopes, fwd, eqn)):
      regions[region] = ()
      eqn_region[eqn] = region

  # Do one pass of forward propagation
  def region_key(region):
    mubatch_idx = 99999 if region.mubatch_idx is None else region.mubatch_idx
    stage = region.stage
    return (mubatch_idx, stage.fwd, stage.kind, stage.index)

  def propagate(eqns, eqn_choices):
    for eqn in eqns:
      if (
        eqn not in eqn_region and
        (region := max(eqn_choices(eqn), default=None, key=region_key))
      ):
        eqn_region[eqn] = region

  var_defn, var_uses = var_defn_and_uses(main_jaxpr)
  # Forward propagation
  propagate(
    main_jaxpr.eqns,
    lambda eqn: (
      region
      for invar in vars_only(eqn.invars)
      if (defn := var_defn.get(invar, None)) and
         (region := eqn_region.get(defn, None))
    ),
  )
  # # Backward propagation
  # propagate(
  #   reversed(main_jaxpr.eqns),
  #   lambda eqn: (
  #     region
  #     for outvar in eqn.outvars
  #     for use in var_uses[outvar]
  #     if (region := eqn_region.get(use, None))
  #   ),
  # )

  # Ensure that replicated equations do not depend on assigned ones
  var_defn: dict[Var, JaxprEqn] = {}
  for eqn in main_jaxpr.eqns:
    # Replicate unassigned equations, but fail if they have any dependency on
    # non-replicated equations. Note that replicated equations may be DCE-d
    # later.
    if eqn_region[eqn] is None:
      print(f'[infer_regions] Warning: replicating unassigned eqn {eqn}')
      for invar in vars_only(eqn.invars):
        if (defn := var_defn.get(invar)):
          if eqn_region[defn] is not None:
            raise ValueError(
              f'[infer_regions] Error: replicated eqn {eqn} @ ' \
              f'{eqn.source_info.traceback} takes input from a ' \
              f'non-replicated equation: {defn} @ {eqn.source_info.traceback}'
            )
    for outvar in eqn.outvars:
      var_defn[outvar] = eqn

  return list(regions.keys()), eqn_region


def dump_inferred_regions(
    closed_jaxpr,
    all_regions,
    eqn_region,
    show_name_stack=False
):
  """Print jaxpr annotated with inferred regions."""
  colors = Colors.BG_COLORS
  def region_color_and_label(eqn):
    if (region := eqn_region[eqn]):
      color = colors[all_regions.index(region) % len(colors)]
      label = region.label
    else:
      color = Colors.BG_BLACK
      label = 'R '  # Replicated
    if show_name_stack:
      names = (x.name for x in eqn.source_info.name_stack.stack)
      label = f"{'/'.join(names):>13} {label}"
    return color, label
  region_color_coded = annotate_eqns(
      region_color_and_label,
      mode='prefix_with_label',
  )
  with jaxpr_custom_print(region_color_coded):
    print(closed_jaxpr)


### Extract task graph given stages

TaskId = simplempmd.TaskId
ArrayRef = simplempmd.ArrayRef[TaskId]
UVar = SVar = Var    # Var in the original, unstaged jaxpr versus after
UAtom = SAtom = Atom # Atom in the original, unstaged jaxpr versus after


def vars_only(atoms: Iterable[Atom]) -> Iterable[Var]:
  return filter(lambda v: isinstance(v, Var), atoms)


@dataclass(frozen=True)
class JaxprPartition:
  """The part of a jaxpr specific to one stage, but without any translation."""
  invar_defn: dict[UVar, MaybeRegion]
  outvar_uses: dict[UVar, set[MaybeRegion]]
  eqns: list[JaxprEqn]

  @property
  def invars(self):
    return self.invar_defn.keys()

  @property
  def outvars(self):
    return self.outvar_uses.keys()


def jaxpr_partitions(
    jaxpr: Jaxpr,
    all_regions: list[Region],
    eqn_region: dict[JaxprEqn, MaybeRegion],
) -> tuple[dict[Region, JaxprPartition], dict[UVar, MaybeRegion]]:
  def eqn_regions(eqn: JaxprEqn) -> tuple[Iterable[Region], bool]:
    """An explicit list of all the regions the given equation is part of."""
    region = eqn_region[eqn]
    if region is None:
      return all_regions, True
    return (region,), False

  var_defn_region: dict[UVar, MaybeRegion] = {
    invar: None for invar in jaxpr.invars
  }
  replicated_vars: set[UVar] = set()
  partitions: dict[Region, JaxprPartition] = {
    region: JaxprPartition(
      invar_defn={},
      outvar_uses=defaultdict(set),
      eqns=[],
    )
    for region in all_regions
  }

  for eqn in jaxpr.eqns:
    regions, is_replicated = eqn_regions(eqn)
    for i, region in enumerate(regions):
      # Register this equation as being part of the region
      partitions[region].eqns.append(eqn)
      # Register dependencies among regions
      for invar in vars_only(eqn.invars):
        defn_region = var_defn_region[invar]
        if defn_region != region and invar not in replicated_vars:
          partitions[region].invar_defn[invar] = defn_region
        if defn_region:
          partitions[defn_region].outvar_uses[invar].add(region)
      # Register defining region for each outvar
      # (Assign outvars of replicated eqns to the first region)
      if i == 0:
        for outvar in eqn.outvars:
          var_defn_region[outvar] = region
          if is_replicated:
            replicated_vars.add(outvar)

  for outvar in vars_only(jaxpr.outvars):
    defn_region = var_defn_region[outvar]
    if defn_region:
      partitions[defn_region].outvar_uses[outvar].add(None)

  assert set(partitions.keys()) == set(all_regions)
  return partitions, var_defn_region


def extract_task_graph(
    closed_jaxpr: ClosedJaxpr,
    all_regions: list[Region],
    eqn_region: dict[JaxprEqn, MaybeRegion],
) -> simplempmd.TaskGraph:
  jaxpr = closed_jaxpr.jaxpr
  constvars_set: set[UVar] = set(jaxpr.constvars)
  constvar_value: dict[UVar, Any] = dict(
    zip(closed_jaxpr.consts, jaxpr.constvars))

  # First break the jaxpr up into one partition per region
  partitions, var_defn_region = jaxpr_partitions(jaxpr, all_regions, eqn_region)

  def extract_array_ref(invar: UVar) -> ArrayRef:
    # FIXME: Slow lookups
    if (defn_region := var_defn_region[invar]):
      task_id = all_regions.index(defn_region)
      return ArrayRef.output(
        task_id, list(partitions[defn_region].outvars).index(invar))
    return ArrayRef.param(jaxpr.invars.index(invar))

  # Then generate a Task for every region
  # TODO: Check that partitions form a DAG and all_regions is toposorted wrt it.
  tasks = []
  for region in all_regions:
    partition = partitions[region]

    # Map UVars to separate SVars for each region
    var_map: dict[UVar, SVar] = {}
    def translate(uvar: UAtom) -> SAtom:
      if isinstance(uvar, Literal):
        return uvar
      if uvar not in var_map:
        # Generate a fresh variable and associate it
        var_map[uvar] = Var(uvar.suffix, uvar.aval)
      return var_map[uvar]

    # Copy the jaxpr while translating all of the variables
    task_eqns = [
      eqn.replace(
        invars=[translate(invar) for invar in eqn.invars],
        outvars=[translate(outvar) for outvar in eqn.outvars],
      )
      for eqn in partition.eqns
    ]
    task_consts = [constvar_value[invar] for invar in partition.invars if invar in constvars_set]
    task_constvars = [translate(invar) for invar in partition.invars if invar in constvars_set]
    task_invars = [translate(invar) for invar in partition.invars if invar not in constvars_set]
    task_outvars = [translate(outvar) for outvar in partition.outvars]
    task_jaxpr = jaxpr.replace(
      constvars=task_constvars,
      invars=task_invars,
      outvars=task_outvars,
      eqns=task_eqns,
      # TODO: Adjust effects and debug_info?
      debug_info=None,
    )
    task_inputs = [extract_array_ref(invar) for invar in partition.invars if invar not in constvars_set]
    # Eliminate dead code and subsequently unused inputs
    task_jaxpr, used_invars = pe.dce_jaxpr(task_jaxpr, (True,) * len(task_outvars))
    task_inputs = [ref for ref, used in zip(task_inputs, used_invars) if used]
    # Package it all up as a task
    task_closed_jaxpr = ClosedJaxpr(consts=task_consts, jaxpr=task_jaxpr)
    tasks.append(simplempmd.Task(
      id=len(tasks),
      inputs=task_inputs,
      num_outputs=len(partition.outvars),
      computation=jax.jit(jax.core.jaxpr_as_fun(task_closed_jaxpr)),
      label=region.label,
      userdata=region,
    ))

  # Extract results
  def extract_result(outvar: UAtom) -> ArrayRef:
    assert isinstance(outvar, Var), f'result {outvar} must be a Var'
    return extract_array_ref(outvar)
  results = [extract_result(outvar) for outvar in jaxpr.outvars]

  return simplempmd.TaskGraph(
    tasks=tasks,
    num_params=len(jaxpr.invars),
    results=results,
  )


### Generic jaxpr transformations

def abstract_vals(vals):
  return list(map(jax.core.get_aval, vals))


def deconstruct_to_jaxpr(fun, dummy_args):
  # TODO: Make this JIT-able (drop dummy_args, take transform_fun callback).
  flat_dummy_args, in_tree = tree_flatten(dummy_args)
  in_paths = [p for p, _ in tree_leaves_with_path(dummy_args)]
  dummy_avals = abstract_vals(flat_dummy_args)

  fun1 = lu.wrap_init(fun)
  fun2, out_tree_thunk = flatten_fun_nokwargs(fun1, in_tree)
  closed_jaxpr = jax.make_jaxpr(fun2.call_wrapped)(*flat_dummy_args)
  out_tree = out_tree_thunk()

  def reconstruct(new_fun):
    @wraps(new_fun)
    def transformed_fun(*args):
      flat_args, in_tree_ = tree_flatten(args)
      assert in_tree_ == in_tree
      assert abstract_vals(flat_args) == dummy_avals
      out_flat = new_fun(*flat_args)
      return tree_unflatten(out_tree, out_flat)
    return transformed_fun

  return reconstruct, closed_jaxpr, in_paths, dummy_avals


### Helpers for simplepp.pipeline

def color_task_graph_edges(task1, task2):
  if not (task1 and task2):
    return {}
  s1: Stage = task1.userdata.stage
  s2: Stage = task2.userdata.stage
  color = 'black'
  if s1.kind == s2.kind == StageKind.Model and s1.fwd and not s2.fwd:
    color = 'red' # stashed activations
  elif s1.kind == s2.kind == StageKind.Model and not s1.fwd and not s2.fwd:
    if s1.index == s2.index:
      color = 'green2' # partially-accumulated grads
    elif s1.index == s2.index + 1:
      color = 'darkmagenta' # grads between stages
  elif s1.kind == StageKind.Model and s2.kind == StageKind.Update and not s1.fwd:
    color = 'green4' # fully-accumulated grads
  return {'color': color}


def mubatched(process_mubatch, num_mubatch, process_only_one):
  def process_batch(batch):
    acc_init = False
    assert all(x.shape[0] == num_mubatch for x in tree_leaves(batch))
    num_iterations = 1 if process_only_one else num_mubatch
    for mubatch_idx in range(num_iterations):
      with mubatch_scope(mubatch_idx):
        mubatch = tree_map(lambda x: x[mubatch_idx], batch)
        # Process current mubatch, and accumulate the results.
        # Only the grads are summed, all other outputs are stacked.
        to_stack, grad = process_mubatch(mubatch)
        if not acc_init:
          acc_stacked = tree_map(
            lambda x: jax.numpy.zeros_like(x, shape=(num_mubatch, *x.shape)),
            to_stack,
          )
          acc_grad = tree_map(jax.numpy.zeros_like, grad)
          acc_init = True
        acc_stacked = tree_map(
          lambda acc, x: jax.lax.dynamic_update_index_in_dim(
            acc, x, index=mubatch_idx, axis=0),
          acc_stacked,
          to_stack,
        )
        acc_grad = tree_map(jax.lax.add, acc_grad, grad)
    return acc_stacked, acc_grad
  return process_batch


### Public API for simplepp

def stage_scope(kind: StageKind, index: Optional[int]=None):
  return jax.named_scope(stage_scope_name(kind, index))


def mubatch_scope(index: int):
  return jax.named_scope(mubatch_scope_name(index))


def pipeline(step_fun, dummy_args, num_mubatch, num_islands):
  mubatcher = partial(mubatched, num_mubatch=num_mubatch, process_only_one=False)
  mubatched_step_fun = partial(step_fun, mubatcher)
  reconstruct, closed_jaxpr, _, _ = deconstruct_to_jaxpr(mubatched_step_fun, dummy_args)

  # Use the scope stack to determine the region of each equation
  all_regions, eqn_region = infer_regions(closed_jaxpr)
  dump_inferred_regions(closed_jaxpr, all_regions, eqn_region)

  # Generate the the microbatched task graph
  task_graph = extract_task_graph(closed_jaxpr, all_regions, eqn_region)
  # print(task_graph)
  simplempmd.dump_task_graph('task_graph', task_graph, edge_kwargs=color_task_graph_edges)

  # Schedule the pipelined task graph
  # NOTE: This is incomplete, i.e., for now we simply emit schedules w/o any parallelism.
  def task_assigner(task_graph, topology):
    assignment = []
    for task in task_graph.tasks:
      stage = task.userdata.stage
      if stage.kind == StageKind.Loss:
        assignment.append((task.id, topology.num_islands - 1))
      else:
        assignment.append((task.id, max(0, stage.index - 1)))
    return assignment

  topology = simplempmd.Topology(num_islands=num_islands)
  program = simplempmd.schedule_tasks(
    task_graph=task_graph,
    topology=topology,
    task_assigner=task_assigner,
  )
  # import pprint; print(f'Scheduled tasks:'); pprint.pprint(program, width=120)
  simplempmd.dump_scheduled_tasks(program)

  batch_fun = reconstruct(lambda *args: simplempmd.execute_local(program, args))
  return batch_fun


if __name__ == '__main__':
  # jax.config.update("jax_log_compiles", "1")
  # jax.config.update("jax_explain_cache_misses", "1")

  @jax.jit
  def g(x):
    return 2 * x

  def step(mubatcher, x):
    def f(x):
      with stage_scope(StageKind.Model, 0):
        g1 = g(x[0]['foo'])
      with stage_scope(StageKind.Model, 1):
        g2 = g(g1)
      with stage_scope(StageKind.Model, 2):
        g3 = g(g2)
      with stage_scope(StageKind.Loss):
        return g3, g3 / x[1]['bar']
    return mubatcher(f)(x)

  M = 2
  ones = jax.numpy.ones((M, 4))
  f_input = ({'foo': ones * 1.0}, {'bar': ones * 2.0})
  f_staged = pipeline(step, dummy_args=(f_input,), num_mubatch=M, num_islands=2)
  f_input2 = ({'foo': ones * 2.0}, {'bar': ones * 3.0})
  print('=== 1 ===')
  print(f_staged(f_input))
  print('=== 2 ===')
  # Doesn't trigger any additional compilation.
  print(f_staged(f_input2))
  print(f_staged(f_input))
