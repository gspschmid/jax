# simplepp

A strawman of what the JAX frontend for "true-MPMD" pipeline-parallelism might look like. Includes a user-facing API to annotate stages, a transformation that introduces microbatching, and a simplistic MPMD runtime that allows executing the resulting task graphs locally.

See `example_mlp.py` for example usage on a simple MLP:
```
python3 example_mlp.py
(...)
```
This will
- Print the step function's jaxpr annotated with stages ([example](img/jaxpr_annotated_snippet.png)).
- Lower the program to an explicit MPMD task graph with #microbatches copies of all stages (e.g. [4 stages, 1 microbatch](img/task_graph.4stages_1mubatch.png) and [4 stages, 8 microbatches](img/task_graph.4stages_8mubatch.png)).
- Interpret that task graph locally, executing the original model in micro-batched manner.
