from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering

from typing import Any, Callable, Generic, Iterable, NewType, Optional, Sequence, TypeVar

import jax
from jax.core import ClosedJaxpr
from jax.util import safe_zip


T = TypeVar('T')


IslandId = NewType("IslandId", int)
TaskId = NewType("TaskId", int)


@dataclass(frozen=True)
class Topology:
  # Each island (logically) runs one SPMD program
  num_islands: int

  @property
  def islands(self) -> Iterable[IslandId]:
    return range(self.num_islands)


@total_ordering
@dataclass(frozen=True)
class ArrayRef(Generic[T]):
  """Refers to a program parameter or task/op output."""
  source: Optional[T]
  index: int

  @property
  def is_param(self):
    """Whether this ArrayRef refers to a program parameter."""
    return self.source is None

  @staticmethod
  def param(index: int) -> 'ArrayRef[T]':
    return ArrayRef(None, index)

  @staticmethod
  def output(source: T, index: int) -> 'ArrayRef[T]':
    return ArrayRef(source, index)

  def __lt__(self, other):
    if self.source != other.source:
      if self.source is None:
        return True
      elif other.source is None:
        return False
      return self.source < other.source
    return self.index < other.index


@dataclass(frozen=True)
class Sharding:
  island: Optional[IslandId]

  @property
  def is_replicated(self):
    return self.island is None

  @staticmethod
  def replicated() -> 'Sharding':
    return Sharding(None)


@dataclass(frozen=True)
class Task:
  id: TaskId
  inputs: list[ArrayRef[TaskId]]
  num_outputs: int
  # Underlying jax.jitted function to be called with "flat" inputs and outputs.
  computation: Callable[[list[jax.Array]], list[jax.Array]]

  label: Optional[str] = None
  userdata: Any = None

  @property
  def outputs(self) -> list[ArrayRef[TaskId]]:
    return [ArrayRef.output(self.id, index) for index in range(self.num_outputs)]


@dataclass(frozen=True)
class TaskGraph:
  tasks: list[Task]
  num_params: int
  results: list[ArrayRef[TaskId]]

  @property
  def task_ids(self) -> list[TaskId]:
    return [task.id for task in self.tasks]

  @property
  def params(self) -> Iterable[ArrayRef[TaskId]]:
    return [ArrayRef.param(index) for index in range(self.num_params)]

  def __post_init__(self):
    assert len(self.task_ids) == len(self.tasks), \
      'all task ids must be distinct'
    assert all(i == task.id for i, task in enumerate(self.tasks)), \
      'tasks should be numbered contiguously and ascending from 0'

    # Tasks must be ordered topologically
    available_refs = set(self.params)
    for task in self.tasks:
      for ref in task.inputs:
        assert ref in available_refs, f'{task=} refers to later task via {ref=}'
      available_refs.update(task.outputs)


OpId = NewType("OpId", int)
OpKind = Enum("OpKind", "Call Send Recv")


@dataclass(frozen=True)
class Op:
  id: OpId
  kind: OpKind

  # Only valid for OpKind.Call
  called_task_id: Optional[TaskId]
  # Only valid for OpKind.Send and OpKind.Recv
  comm_peer: Optional[IslandId]

  inputs: list[ArrayRef[OpId]]
  num_outputs: int

  @property
  def outputs(self) -> list[ArrayRef[OpId]]:
    return [ArrayRef.output(self.id, idx) for idx in range(self.num_outputs)]

  @staticmethod
  def call(id: OpId, called_task_id: TaskId, inputs: list[ArrayRef[OpId]], num_outputs: int) -> 'Op':
    return Op(id, OpKind.Call, called_task_id, None, inputs, num_outputs)

  @staticmethod
  def send(id: OpId, comm_peer: IslandId, inputs: list[ArrayRef[OpId]]) -> 'Op':
    return Op(id, OpKind.Send, None, comm_peer, inputs, num_outputs=0)

  @staticmethod
  def recv(id: OpId, comm_peer: IslandId, num_outputs: int) -> 'Op':
    return Op(id, OpKind.Recv, None, comm_peer, [], num_outputs)


@dataclass(frozen=True)
class Program:
  """A program results from scheduling a task graph on a topology.
  Task invocations and communication are represented as explicit ops.
  """
  task_graph: TaskGraph
  topology: Topology
  param_shardings: list[Sharding]
  result_shardings: list[Sharding]
  results: list[tuple[IslandId, ArrayRef[OpId]]]
  ops_by_island: list[list[Op]]

  def ops(self) -> Iterable[tuple[IslandId, Op]]:
    for island, ops in enumerate(self.ops_by_island):
      for op in ops:
        yield (island, op)

  @property
  def params(self) -> Iterable[ArrayRef[OpId]]:
    return [ArrayRef.param(index) for index in range(len(self.param_shardings))]

  def __post_init__(self):
    task_graph = self.task_graph
    assert len(self.ops_by_island) == len(self.topology.islands)
    assert len(self.param_shardings) == task_graph.num_params
    assert len(self.result_shardings) == len(task_graph.results)
  
    # Every task is scheduled exactly once
    task_ids_called = [
      op.called_task_id for _, op in self.ops() if op.kind == OpKind.Call
    ]
    assert set(task_ids_called) == set(task_graph.task_ids), \
      'every task must be scheduled exactly once'
    assert len(task_ids_called) == len(task_graph.tasks), \
      'no task may be called more than once'

    # Program respects task dependencies and doesn't deadlock
    trace = []
    available_refs_by_island = [set(self.params) for _ in self.topology.islands]

    def handle_op(island, op):
      trace.append(island)
      available_refs = available_refs_by_island[island]
      for ref in op.inputs:
        assert ref in available_refs, \
          f'{island=} {op=} relies {ref=} that is not available yet'
      available_refs.update(op.outputs)

    deadlock = interpret(self, handle_op)
    assert not deadlock, f'deadlock in schedule: {trace=}'


def interpret(
    program: Program,
    handle_op: Callable[[IslandId, Op], Iterable[IslandId]],
) -> bool:
  op_stacks = [list(reversed(ops)) for ops in program.ops_by_island]
  barrier_arrivals = [[0 for _ in op_stacks] for _ in op_stacks]
  while True:
    for island, op_stack in enumerate(op_stacks):
      if not op_stack:
        # Island has completed all of its ops
        continue
      op = op_stack[-1]
      if op.kind == OpKind.Send:
        if barrier_arrivals[island][op.comm_peer] % 2 == 1:
          # Blocked on send, because the peer has another outstanding recv
          continue
        barrier_arrivals[island][op.comm_peer] += 1
      elif op.kind == OpKind.Recv:
        if barrier_arrivals[op.comm_peer][island] % 2 == 0:
          # Blocked on recv, because the peer hasn't sent yet
          continue
        barrier_arrivals[op.comm_peer][island] += 1
      # Island made some progess
      op_stack.pop()
      handle_op(island, op)
      break
    else:
      # Cannot make any more progress
      deadlock = bool(any(op_stacks))
      return deadlock


TaskAssignment = list[tuple[TaskId, IslandId]]
TaskAssignmentFn = Callable[[TaskGraph, Topology], TaskAssignment]

def round_robin(task_graph: TaskGraph, topology: Topology) -> TaskAssignment:
  n = len(topology.islands)
  return [(i, i % n) for i in range(len(task_graph.tasks))]


def schedule_tasks(
    task_graph: TaskGraph,
    topology: Topology,
    task_assigner: TaskAssignmentFn,
) -> Program:
  task_assignment = task_assigner(task_graph, topology)
  task_island: dict[TaskId, IslandId] = dict(task_assignment)

  uses: dict[ArrayRef[TaskId], list[TaskId]] = defaultdict(list)
  for task in task_graph.tasks:
    for input in task.inputs:
      uses[input].append(task.id)

  # Lower to ops for each island
  ops_by_island: list[list[Op]] = [[] for _ in topology.islands]
  op_refs_by_island: list[dict[ArrayRef[TaskId], ArrayRef[OpId]]] = [
    {ref: ArrayRef.param(ref.index) for ref in task_graph.params}
    for island in topology.islands
  ]

  def add_op(island: IslandId, task_outputs, op_builder, *args, **kwargs):
    if task_outputs is not None:
      kwargs['num_outputs'] = len(task_outputs)
    else:
      task_outputs = []
    new_op_id = len(ops_by_island[island])
    op = op_builder(new_op_id, *args, **kwargs)
    ops_by_island[island].append(op)
    for task_output, op_output in safe_zip(task_outputs, op.outputs):
      op_refs_by_island[island][task_output] = op_output

  for task_id, island in task_assignment:
    task = task_graph.tasks[task_id]
    ops = ops_by_island[island]
    op_refs = op_refs_by_island[island]

    # Schedule the computation
    inputs = [op_refs[input_ref] for input_ref in task.inputs]
    add_op(island, task.outputs, Op.call, task.id, inputs)

    # Schedule the communication
    for ref in task.outputs:
      # TODO: Group sends of multiple arrays between the same islands
      user_islands = set(task_island[user_task_id] for user_task_id in uses[ref])
      for user_island in sorted(user_islands):
        if user_island == island:
          continue
        add_op(island, None, Op.send, user_island, [op_refs[ref]])
        add_op(user_island, [ref], Op.recv, island)

  # Infer the program's input and output shardings
  def infer_sharding(ref: ArrayRef[TaskId]) -> Sharding:
    islands = set(task_island[user_task_id] for user_task_id in uses[ref])
    if len(islands) == 1:
      return Sharding(next(iter(islands)))
    return Sharding.replicated()
  param_shardings = [infer_sharding(ref) for ref in task_graph.params]

  result_shardings = [
    (param_shardings[ref.index]
      if ref.is_param else Sharding(task_island[ref.source]))
    for ref in task_graph.results
  ]
  results = [
    ((island := task_island[ref.source]), op_refs_by_island[island][ref])
    for ref in task_graph.results
  ]

  return Program(
    task_graph=task_graph,
    topology=topology,
    param_shardings=param_shardings,
    result_shardings=result_shardings,
    results=results,
    ops_by_island=ops_by_island,
  )


def execute_local(program: Program, inputs: list[jax.Array]) -> list[jax.Array]:
  assert len(inputs) == len(program.params), 'unexpected number of inputs'
  islands = list(program.topology.islands)
  state_by_island: list[dict[ArrayRef[OpId], jax.Array]] = [
    {} for _ in islands
  ]
  for ref, param_sharding, input in safe_zip(program.params, program.param_shardings, inputs):
    for island in islands:
      if param_sharding.is_replicated or island == param_sharding.island:
        state_by_island[island][ref] = input

  inbox = [[None for _ in islands] for _ in islands]

  def handle_op(island, op):
    state = state_by_island[island]
    in_arrs = [state[ref] for ref in op.inputs]
    if op.kind == OpKind.Call:
      task = program.task_graph.tasks[op.called_task_id]
      try:
        out_arrs = task.computation(*in_arrs)
      except Exception as e:
        print(f'[execute_local]: Exception occurred executing ' \
              f'{island=} {task.label=} {task.id=}')
        raise e
      assert isinstance(out_arrs, list | tuple), \
        f'task computations must return list of outputs, got {type(out_arrs)}'
    elif op.kind == OpKind.Send:
      assert inbox[island][op.comm_peer] is None
      inbox[island][op.comm_peer] = in_arrs[0]
      out_arrs = []
    elif op.kind == OpKind.Recv:
      assert inbox[op.comm_peer][island] is not None
      out_arrs = [inbox[op.comm_peer][island]]
      inbox[op.comm_peer][island] = None
    else:
      assert False
    for ref, out_arr in safe_zip(op.outputs, out_arrs):
      state[ref] = out_arr

  interpret(program, handle_op)
  return [state_by_island[island][op_ref] for island, op_ref in program.results]


def dump_task_graph(
    filename,
    task_graph,
    show_params=False,
    show_results=False,
    edge_kwargs=None,
):
  from itertools import groupby
  from graphviz import Digraph

  start_name = 'Start'
  end_name = 'End'

  def task_name(task_id: TaskId):
    return f'Task{task_id}'
  def input_name(source: Optional[TaskId]):
    return task_name(source) if source is not None else start_name

  graph = Digraph(graph_attr={'rankdir': 'LR'})
  if show_params:
    graph.node(start_name, start_name)
  if show_results:
    graph.node(end_name, end_name)

  def add_in_edges_grouped(in_refs, to_task):
    to_name = end_name if to_task is None else task_name(to_task.id)
    in_refs = sorted(
      in_refs,
      key=lambda ref: ref.source if ref.source is not None else -1)
    edges_by_source = groupby(enumerate(in_refs), lambda t: t[1].source)
    for source, edges in edges_by_source:
      if not show_params and source is None:
        continue
      kwargs = {}
      if edge_kwargs:
        source_task = None if source is None else task_graph.tasks[source]
        kwargs = edge_kwargs(source_task, task)
      if 'label' not in kwargs:
        kwargs['label'] = ', '.join(f'{ref.index}→{i}' for i, ref in edges)
      graph.edge(input_name(source), to_name, **kwargs)

  for task in task_graph.tasks:
    name = task_name(task.id)
    graph.node(name, task.label or name)
    add_in_edges_grouped(task.inputs, task)
  if show_results:
    add_in_edges_grouped(task_graph.results, None)

  graph.render(filename, format='png', cleanup=True)


def dump_scheduled_tasks(program, show_op_edges=False):
  from itertools import groupby
  from graphviz import Digraph

  # IslandId = simplempmd.IslandId
  # OpId = simplempmd.OpId
  # OpKind = simplempmd.OpKind
  islands = program.topology.islands
  tasks = program.task_graph.tasks

  def op_name(island: IslandId, op_id: OpId):
    return f'Op({island},{op_id})'
  def input_name(island: IslandId, source: OpId):
    return op_name(island, source)

  graph = Digraph(graph_attr={'rankdir': 'LR'})
  subgraphs = [
    Digraph(
      name=f'Island{island}',
      graph_attr={'rankdir': 'LR'},
    )
    for island in islands
  ]

  def add_local_edges_grouped(island, in_refs, node_name):
    in_refs = sorted(in_refs, key=lambda ref: ref.source if ref.source is not None else -1)
    edges_by_source = groupby(enumerate(in_refs), lambda t: t[1].source)
    for source, edges in edges_by_source:
      if source is None:
        continue
      input_label = ', '.join(f'{ref.index}→{i}' for i, ref in edges)
      subgraphs[island].edge(input_name(island, source), node_name, input_label)

  queues = [[[] for _ in islands] for _ in islands]
  for island in islands:
    for op in program.ops_by_island[island]:
      name = op_name(island, op.id)
      label = f'{op.kind}'
      shape = 'box'
      if op.kind == OpKind.Call:
        task = tasks[op.called_task_id]
        label = f'Call {task.label or name}'
      elif op.kind == OpKind.Send:
        label = 'Send'
        style = 'dashed'
        shape = 'rarrow'
        queues[island][op.comm_peer].append(name)
      elif op.kind == OpKind.Recv:
        label = 'Recv'
        style = 'dashed'
        shape = 'larrow'
      else:
        assert False
      subgraphs[island].node(name, label, {'shape': shape})
      if show_op_edges:
        add_local_edges_grouped(island, op.inputs, name)

  # Add invisible edges to enforce program order among op nodes
  for island in islands:
    ops = program.ops_by_island[island]
    for op1, op2 in zip(ops, ops[1:]):
      subgraphs[island].edge(
        op_name(island, op1.id),
        op_name(island, op2.id),
        style='invis' if show_op_edges else 'dotted',
        arrowhead='none',
      )

  # Add communication edges and align vertically
  for island1 in islands:
    for island2 in islands:
      queues[island1][island2].reverse()
  for island in islands:
    for op in program.ops_by_island[island]:
      if op.kind == OpKind.Recv:
        peer_op_name = queues[op.comm_peer][island].pop()
        name = op_name(island, op.id)
        with graph.subgraph(graph_attr={'rank': 'same'}) as c:
          c.node(peer_op_name, group=f'group{op.comm_peer}')
          c.node(name, group=f'group{island}')
        graph.edge(
          name,
          peer_op_name,
          style='dashed',
          constraint='false',
          dir='back',
        )

  # Add vertical alignment at the left edge and order by island
  def island_start(island):
    return op_name(island, -1)
  with graph.subgraph(graph_attr={'rank': 'same'}) as c:
    for island in islands:
      c.node(island_start(island), group=f'group{island}', shape='box', label=f'Island {island}', style='dotted')
  for island in islands:
    graph.edge(
      island_start(island),
      op_name(island, program.ops_by_island[island][0].id),
      style='invis',
    )
  for island1, island2 in zip(islands, islands[1:]):
    graph.edge(island_start(island1), island_start(island2), style='invis')

  for subgraph in subgraphs:
    graph.subgraph(subgraph)
  graph.render('scheduled_tasks', format='png', cleanup=True)


if __name__ == '__main__':
  import pprint
  import numpy as np
  import jax.numpy as jnp
  from jax._src import test_util as jtu
  jtu.set_host_platform_device_count(4)

  def f(a, x):
    return x + a,
  
  def g(b, x):
    return b * x,

  reference_fun = lambda a, b, x: g(b, f(a, x)[0])[0]

  # A task graph that corresponds to reference_fun
  task_graph = TaskGraph(
    tasks=[
      Task(
        id=0,
        label='f',
        inputs=[ArrayRef.param(0), ArrayRef.param(2)],
        num_outputs=1,
        computation=f,
      ),
      Task(
        id=1,
        label='g',
        inputs=[ArrayRef.param(1), ArrayRef.output(0, 0)],
        num_outputs=1,
        computation=g,
      ),
    ],
    num_params=3,
    results=[ArrayRef.output(1, 0)],
  )
  topology = Topology(num_islands=2)
  program = schedule_tasks(
    task_graph=task_graph,
    topology=topology,
    task_assigner=round_robin,
  )
  dump_task_graph('task_graph', program.task_graph)
  print(f'Scheduled tasks:'); pprint.pprint(program, width=120)

  a, b, x = jnp.ones((2,)), jnp.ones((2,)), jnp.ones((2,2,))
  inputs = [a, b, x]
  expected_y = reference_fun(*inputs)
  [actual_y] = execute_local(program, inputs)
  np.testing.assert_array_equal(actual_y, expected_y)
  print('Success!')
