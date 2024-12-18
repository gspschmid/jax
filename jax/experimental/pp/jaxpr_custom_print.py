import contextlib
import functools
import dataclasses

from typing import Callable, Sequence

import jax
import jax._src.pretty_printer as pp
from jax.core import Atom, Jaxpr, JaxprEqn, JaxprPpContext, JaxprPpSettings


### Patch jaxpr pretty-printing

def patch_module_function(mod, fun_name, transformation):
  orig_fun = getattr(mod, fun_name)
  setattr(mod, fun_name, functools.partial(transformation, orig_fun))


_jaxpr_custom_print_eqn = None
def _patched_pp_eqn(orig_fun, eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if _jaxpr_custom_print_eqn:
    return _jaxpr_custom_print_eqn(orig_fun, eqn, context, settings)
  else:
    return orig_fun(eqn, context, settings)
patch_module_function(jax._src.core, 'pp_eqn', _patched_pp_eqn)


@contextlib.contextmanager
def jaxpr_custom_print(pp_eqn):
  global _jaxpr_custom_print_eqn
  old = _jaxpr_custom_print_eqn
  try:
    _jaxpr_custom_print_eqn = pp_eqn
    yield
  finally:
    _jaxpr_custom_print_eqn = old


### Color helpers

class Colors:
  RESET       = "\033[0m"
  BOLD        = "\033[1m"

  FG_BLACK    = "\033[30m"
  FG_RED      = "\033[31m"
  FG_GREEN    = "\033[32m"
  FG_YELLOW   = "\033[33m"
  FG_BLUE     = "\033[34m"
  FG_MAGENTA  = "\033[35m"
  FG_CYAN     = "\033[36m"
  FG_WHITE    = "\033[37m"
  FG_GRAY     = "\033[90m"
  FG_RED2     = "\033[91m"
  FG_GREEN2   = "\033[92m"
  FG_YELLOW2  = "\033[93m"
  FG_BLUE2    = "\033[94m"
  FG_MAGENTA2 = "\033[95m"
  FG_CYAN2    = "\033[96m"
  FG_WHITE2   = "\033[97m"

  BG_BLACK    = "\033[40m"
  BG_RED      = "\033[41m"
  BG_GREEN    = "\033[42m"
  BG_YELLOW   = "\033[43m"
  BG_BLUE     = "\033[44m"
  BG_MAGENTA  = "\033[45m"
  BG_CYAN     = "\033[46m"
  BG_WHITE    = "\033[47m"
  BG_GRAY     = "\033[100m"
  BG_RED2     = "\033[101m"
  BG_GREEN2   = "\033[102m"
  BG_YELLOW2  = "\033[103m"
  BG_BLUE2    = "\033[104m"
  BG_MAGENTA2 = "\033[105m"
  BG_CYAN2    = "\033[106m"
  BG_WHITE2   = "\033[107m"

  FG_COLORS = [
    FG_RED, FG_GREEN, FG_YELLOW, FG_BLUE, FG_MAGENTA, FG_CYAN,
    FG_RED2, FG_GREEN2, FG_YELLOW2, FG_BLUE2, FG_MAGENTA2, FG_CYAN2,
  ]
  BG_COLORS = [
    BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_MAGENTA, BG_CYAN,
    BG_RED2, BG_GREEN2, BG_YELLOW2, BG_BLUE2, BG_MAGENTA2, BG_CYAN2,
  ]


Color = str
Label = str


def annotate_eqns(
    eqn_map: Callable[[JaxprEqn], Color | tuple[Color, Label]],
    mode: str = 'prefix',
):
  def pp_eqn(orig_fun, eqn, context, settings):
    orig_res = orig_fun(eqn, context, settings)
    color = eqn_map(eqn)
    label = '?'
    if isinstance(color, tuple):
      color, label = color
    if mode == 'highlight':
      # Highlight in color
      return pp.text(color) + orig_res + pp.text(Colors.RESET)
    elif mode == 'prefix':
      # Prefix one color block
      prefix = pp.text(f'{color} {Colors.RESET} ')
      return prefix + orig_res
    elif mode == 'prefix_with_label':
      # Prefix one color block with label
      prefix = pp.text(f'{color}{Colors.FG_GRAY}{label}{Colors.RESET} ')
      return prefix + orig_res
    else:
      raise ValueError(f'invalid {mode=}')
  return pp_eqn


### Example usage

if __name__ == '__main__':
  @jax.jit
  def g(x):
    return 2 * x

  def f(x):
    g1 = g(x[0]['foo'])
    g2 = g(g1)
    return g2 / x[1]['bar']

  annotated = annotate_eqns(
      eqn_map=lambda eqn: Colors.BG_COLORS[(id(eqn) >> 6) % len(Colors.BG_COLORS)],
      mode='prefix',
  )

  with jaxpr_custom_print(annotated):
    f_input = ({'foo': 1.0}, {'bar': 2.0})
    print(jax.make_jaxpr(f)(f_input))
