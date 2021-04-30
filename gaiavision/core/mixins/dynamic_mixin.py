"""Mixin classes for dynamic modules.

TODO: whether extract `deploy` from `DynamicMixin` to a `DeployMixin`? Any suggestions?
"""
# standard lib
from copy import deepcopy

class DynamicMixin(object):
    """ Mixin defining all the operations to manipulate architecture of a module.

    The function of this mixin includes:
    -- Demonstrate the valid search space of a dynamic module
    -- Demonstrate the current arch state of a dynamic module
    -- Inspired by duck-typing, the behaviour of arch manipulation is determined during runtime, validated by the search space.
    As an trivial example, TODO:
        ...
    """
    @property
    def search_space(self):
        """Return the pre-defined search space of subclass. Note that `search_space`
        should be a class method.  """
        return self.__class__.search_space

    @property
    def arch_state(self):
        search_space = [v.split('_', 1)[-1] + '_state' for v in self.__class__.__dict__.keys() if v.startswith('manipulate_')]
        search_space.remove('arch_state')
        # should not mutate source arch state
        states = deepcopy({k:getattr(self, k) for k in search_space})
        return states

    def validate(self, arch_meta):
        """ `xxx_state` is reserved keys that should match `manipulate_xxx`
        """
        pass

    def manipulate_arch(self, arch_meta):
        manipulate_fmt = 'manipulate_{}'
        assert isinstance(arch_meta, dict), f'`arch_meta` should be a dict,' \
                                            f' got {type(arch_meta)}'
        for k,v in arch_meta.items():
            manipulator = getattr(self, manipulate_fmt.format(k), None)
            if manipulator is not None:
                manipulator(v)
            else:
                raise Exception(f'`{k}` is not a supported operand of manipulator'
                                f' for `{self.__class__.__name__}`')

    # def deploy(self, arch_meta=None):
    #     # TODO: replace the workaround
    #     manipulable_fields = set([v.split('_', 1)[1] for v in self.__dict__.keys() \
    #                                 if v.startswith('manipulate_')])
    #     state_fields = set([v.rsplit('_', 1)[0] for v in self.__dict__.keys() \
    #                         if v.endswith('_state')])
    #     deploy_fields = manipulable_fields.intersection(state_fields)
    #     if arch_meta is not None:
    #         raise NotImplementedError
    #     else:



