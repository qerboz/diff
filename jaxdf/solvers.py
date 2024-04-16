from typing import Any, Callable
from functools import partial

import chex
import diffrax
import flax.struct
import jax


Integrator = Callable[[Any, chex.Scalar, chex.Scalar, chex.Scalar, Any, Any], Any]


@flax.struct.dataclass
class DiffraxIntegrator:
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()
    adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()
    discrete_terminating_event: diffrax.AbstractDiscreteTerminatingEvent = None
    max_steps: int = 16 ** 3

    def __call__(self, field, t0, t1, dt0, initial, args):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(field),
            self.solver,
            t0, t1, dt0, initial, args,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            discrete_terminating_event=self.discrete_terminating_event,
            max_steps=self.max_steps,
        )
        x, log_prob = sol.ys
        return x[-1], log_prob[-1]
    
def _solver(fn, integrator):
    def inner(scope_fun, repack_fun, variable_groups, rng_groups,
              t0, t1, dt0, y0, args=None):

        @partial(jax.jit, static_argnames=('return_scope',))
        def vf(t, x, a, return_scope=False):
            scope = scope_fun(variable_groups, rng_groups)
            y = fn(scope, t, x, a)
            return repack_fun(scope) if return_scope else y

        return integrator(vf, t0, t1, dt0, y0, args), vf(t0, y0, args, True)

    return flax.core.lift.pack(
        inner, (True,), (True,), (True,), name='ode_solver')