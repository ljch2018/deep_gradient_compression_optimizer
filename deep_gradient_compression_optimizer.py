# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper optimizer for Deep Gradient Compression."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook

import numpy as np

class DeepGradientCompressionCustomGetter(object):

    def __init__(self, worker_device):
        self._worker_device = worker_device
        self._var_2_velocity = {}
        self._var_2_residual = {}

    def __call__(self, getter, name, trainable, collections, *args, **kwargs):
        if trainable:
            global_var = getter(name=name,
                                trainable=True,
                                collections=[ops.GraphKeys.GLOBAL_VARIABLES],
                                *args,
                                **kwargs)
            with ops.device(self._worker_device):
                velocity = variable_scope.variable(name="%s/velocity" % (name),
                                                   initial_value=array_ops.zeros(array_ops.shape(global_var)),
                                                   trainable=False,
                                                   collections=[ops.GraphKeys.LOCAL_VARIABLES])
                residual = variable_scope.variable(name="%s/residual" % (name),
                                                   initial_value=array_ops.zeros(array_ops.shape(global_var)),
                                                   trainable=False,
                                                   collections=[ops.GraphKeys.LOCAL_VARIABLES])

            self._var_2_velocity[global_var] = velocity
            self._var_2_residual[global_var] = residual
            return global_var
        else:
            return getter(name, trainable, collections, *args, **kwargs)


class DeepGradientCompressionOptimizer(optimizer.Optimizer):

    def __init__(self,
                 opt,
                 momentum,
                 replicas_to_aggregate,
                 dgc_custom_getter,
                 total_num_replicas=None,
                 variable_averages=None,
                 variables_to_average=None,
                 use_locking=False,
                 name="DeepGradientCompressionOptimier",
                 use_nesterov=False
                 ):
        if total_num_replicas is None:
            total_num_replicas = replicas_to_aggregate

        super(DeepGradientCompressionOptimizer, self).__init__(use_locking, name)
        self._opt = opt
        self._momentum = momentum
        self._replicas_to_aggregate = replicas_to_aggregate
        self._var_2_velocity = dgc_custom_getter._var_2_velocity
        self._var_2_residual = dgc_custom_getter._var_2_residual
        self._total_num_replicas = total_num_replicas
        self._variable_averages = variable_averages
        self._variables_to_average = variables_to_average
        self._use_locking = use_locking
        self._name = name
        self._use_nesterov = use_nesterov
        self._gradients_applied = False
        self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate)
        self._global_step = None
        self._sync_token_queue = None

        self._chief_queue_runner = None

        self._accumulator_list = []

    def compute_gradient(self, *args, **kwargs):
        return self._opt.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if not grads_and_vars:
            raise ValueError("Must supply at least one variable")

        if global_step is None:
            raise ValueError("Global step is required to check staleness")

        self._global_step = global_step
        train_ops = []
        aggregated_grad = []

        # local_anchor op will be placed on this worker task by default.
        local_anchor = control_flow_ops.no_op()
        # Colocating local_step variable prevents it being placed on the PS.
        with ops.colocate_with(local_anchor):
            self._local_step = variable_scope.variable(
                initial_value=0,
                trainable=False,
                collections=[ops.GraphKeys.LOCAL_VARIABLES],
                dtype=global_step.dtype.base_dtype,
                name="local_step")

        self.local_step_init_op = state_ops.assign(self._local_step, global_step)
        chief_init_ops = [self.local_step_init_op]
        self.ready_for_local_init_op = variables.report_uninitialized_variables(
            variables.global_variables())

        var_list = [v for g, v in grads_and_vars]
        velocity_list = [self._var_2_velocity[v] for v in var_list]
        residual_list = [self._var_2_residual[v] for v in var_list]

        density = 0.01

        with ops.name_scope(None, self._name):
            for velocity, residual, grad, var in zip(velocity_list, residual_list, grads_and_vars):
                if grad is not None:
                    if self._use_nesterov:
                        update_velocity = self._momentum * (velocity + grad)
                        update_residual = residual + update_velocity + grad
                    else:
                        update_velocity = self._momentum * velocity + grad
                        update_residual = residual + update_velocity
                else:
                    update_velocity = velocity
                    update_residual = residual

                # select threshold according to abs(update_residual)
                top_k_values, top_k_indices = nn_ops.top_k(math_ops.abs(update_residual),
                                                    math_ops.to_int32(array_ops.shape(update_residual)[-1] * density))
                threshold = top_k_values[-1]
                mask = math_ops.abs(update_residual) > threshold
                mask = math_ops.cast(mask, dtype = dtypes.int32)
                mask_h = math_ops.abs(mask - 1)

                with ops.device(grad.device):
                    dense_grad = mask * update_residual
                    indices = array_ops.where(math_ops.not_equal(dense_grad, 0))
                    values = array_ops.gather_nd(dense_grad, indices)
                    sparse_grad = ops.IndexedSlices(values, indices, dense_grad.get_shape())
                    #grad_update = state_ops.assign(grad, mask * update_residual)

                #with ops.control_dependencies([grad_update]), ops.device(var.device):
                    #grad_accum = data_flow_ops.ConditionalAccumulator(
                        #grad.dtype, shape=var.get_shape(),
                        #shared_name=var.name + "/grad_accum")
                    #train_ops.append(grad_accum.apply_grad(grad, local_step=self._local_step))
                    #aggregated_grad.append(grad_accum.take_grad(self._replicas_to_aggregate))

                with ops.device(var.device):
                    grad_accum = data_flow_ops.SparseConditionalAccumulator(
                        sparse_grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
                    train_ops.append(grad_accum.apply_indexed_slices_grad(
                        sparse_grad, local_step=self._local_step))
                    aggregated_grad.append(grad_accum.take_indexed_slices_grad(self._replicas_to_aggregate))

                    self._accumulator_list.append((grad_accum, var.device))

                with ops.device(residual.device):
                    train_ops.append(state_ops.assign(residual, mask_h * update_residual))
                with ops.device(velocity.device):
                    train_ops.append(state_ops.assign(velocity, mask_h * update_velocity))

            aggregated_grads_and_vars = zip(aggregated_grad, var_list)

            with ops.device(global_step.device), ops.name_scope(""):
                update_op = self._opt.apply_gradient(aggregated_grads_and_vars, global_step)


            with ops.device(global_step.device), ops.name_scope(""):
                sync_token_queue = (
                    data_flow_ops.FIFOQueue(-1,
                                            global_step.dtype.base_dtype,
                                            shapes=(),
                                            name="sync_token_q",
                                            shared_name="sync_token_q"))
                self._sync_token_queue = sync_token_queue

                dummy_queue = (
                    data_flow_ops.FIFOQueue(1,
                                            types_pb2.DT_INT32,
                                            shapes=(),
                                            name="dummy_queue",
                                            shared_name="dummy_queue"))

                with ops.control_dependencies(train_ops):
                    token = sync_token_queue.dequeue()
                train_op = state_ops.assign(self._local_step, token)

                with ops.control_dependencies([update_op]):
                    tokens = array_ops.fill([self._tokens_per_step], global_step)
                    sync_op = sync_token_queue.enqueue_many((tokens,))

                if self._variable_averages is not None:
                    with ops.control_dependencies([sync_op]), ops.name_scope(""):
                        sync_op = self._variable_averages.apply(self._variables_to_average)

                self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue, [sync_op])

            for accum, dev in self._accumulator_list:
                with ops.device(dev):
                    chief_init_ops.append(accum.set_global_step(global_step, name="SetGlobalStep"))
            self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
            self._gradients_applied = True

            return train_op

    def get_chief_queue_runner(self):
        """Returns the QueueRunner for the chief to execute.

        This includes the operations to synchronize replicas: aggregate gradients,
        apply to variables, increment global step, insert tokens to token queue.

        Note that this can only be called after calling apply_gradients() which
        actually generates this queuerunner.

        Returns:
          A `QueueRunner` for chief to execute.

        Raises:
          ValueError: If this is called before apply_gradients().
        """
        if self._gradients_applied is False:
            raise ValueError("Should be called after apply_gradients().")

        return self._chief_queue_runner

    def get_slot(self, *args, **kwargs):
        """Return a slot named "name" created for "var" by the Optimizer.

        This simply wraps the get_slot() from the actual optimizer.

        Args:
          *args: Arguments for get_slot().
          **kwargs: Keyword arguments for get_slot().

        Returns:
          The `Variable` for the slot if it was created, `None` otherwise.
        """
        return self._opt.get_slot(*args, **kwargs)

    def variables(self):
        """Fetches a list of optimizer variables in the default graph.

        This wraps `variables()` from the actual optimizer. It does not include
        the `DeepGradientCompressionOptimizer`'s local step.

        Returns:
          A list of variables.
        """
        return self._opt.variables()

    def get_slot_names(self, *args, **kwargs):
        """Return a list of the names of slots created by the `Optimizer`.

        This simply wraps the get_slot_names() from the actual optimizer.

        Args:
          *args: Arguments for get_slot().
          **kwargs: Keyword arguments for get_slot().

        Returns:
          A list of strings.
        """
        return self._opt.get_slot_names(*args, **kwargs)

    def get_init_tokens_op(self, num_tokens=-1):
        """Returns the op to fill the sync_token_queue with the tokens.

        This is supposed to be executed in the beginning of the chief/sync thread
        so that even if the total_num_replicas is less than replicas_to_aggregate,
        the model can still proceed as the replicas can compute multiple steps per
        variable update. Make sure:
        `num_tokens >= replicas_to_aggregate - total_num_replicas`.

        Args:
          num_tokens: Number of tokens to add to the queue.

        Returns:
          An op for the chief/sync replica to fill the token queue.

        Raises:
          ValueError: If this is called before apply_gradients().
          ValueError: If num_tokens are smaller than replicas_to_aggregate -
            total_num_replicas.
        """
        if self._gradients_applied is False:
            raise ValueError(
                "get_init_tokens_op() should be called after apply_gradients().")

        tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
        if num_tokens == -1:
            num_tokens = self._replicas_to_aggregate
        elif num_tokens < tokens_needed:
            raise ValueError(
                "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
                (num_tokens, tokens_needed))

        if num_tokens > 0:
            with ops.device(self._global_step.device), ops.name_scope(""):
                tokens = array_ops.fill([num_tokens], self._global_step)
                init_tokens = self._sync_token_queue.enqueue_many((tokens,))
        else:
            init_tokens = control_flow_ops.no_op(name="no_init_tokens")

        return init_tokens

    def make_session_run_hook(self, is_chief, num_tokens=-1):
        """Creates a hook to handle DeepGradientCompressionHook ops such as initialization."""
        return _DeepGradientCompressionOptimizerHook(self, is_chief, num_tokens)

class _DeepGradientCompressionOptimizerHook(session_run_hook.SessionRunHook):
    """A SessionRunHook handles ops related to DeepGradientCompressionOptimizer."""

    def __init__(self, dgc_optimizer, is_chief, num_tokens):
        self._dgc_optimizer = dgc_optimizer
        self._is_chief = is_chief
        self._num_tokens = num_tokens

    def begin(self):
        if self._dgc_optimizer._gradient_applied is False:
            raise ValueError(
                "DeepGradientCompressionOptimizer.apply_gradient should be called before using "
                "the hook.")
        if self._is_chief:
            self._local_init_op = self._dgc_optimizer.chief_init_op
            self._ready_for_local_init_op = (
                self._dgc_optimizer.ready_for_local_init_op)
            self._q_runner = self._dgc_optimizer.get_chief_queue_runner()
            self._init_tokens_op = self._dgc_optimizer.get_init_tokens_op(
                self._num_tokens)
        else:
            self._local_init_op = self._dgc_optimizer.local_step_init_op
            self._ready_for_local_init_op = (
                self._dgc_optimizer.ready_for_local_init_op)
            self._q_runner = None
            self._init_tokens_op = None

    def after_create_session(self, session, coord):
        """Runs DeepGradientCompressionOptimizer initialization ops."""
        local_init_success, msg = session_manager._ready(  # pylint: disable=protected-access
            self._ready_for_local_init_op, session,
            "Model is not ready for DeepGradientCompressionOptimizer local init.")
        if not local_init_success:
            raise RuntimeError(
                "Init operations did not make model ready for DeepGradientCompressionOptimizer "
                "local_init. Init op: %s, error: %s" %
                (self._local_init_op.name, msg))
        session.run(self._local_init_op)
        if self._init_tokens_op is not None:
            session.run(self._init_tokens_op)
        if self._q_runner is not None:
            self._q_runner.create_threads(
                session, coord=coord, daemon=True, start=True)









