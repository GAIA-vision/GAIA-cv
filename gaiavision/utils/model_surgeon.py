from abc import ABCMeta, abstractmethod


class BaseSurgeon(metaclass=ABCMeta):
    """ All subclasses should implement the following APIs:
    - operate_on: process the state_dict
    """
    @abstractmethod
    def operate_on(self, state_dict):
        pass


class FCMapLabelSurgeon(BaseSurgeon):
    def __init__(self,
                 label_mapping,
                 fc_name='roi_head.bbox_head.fc_cls',
                 random_map=False,
                 class_agnostic=True):
        self.label_mapping = label_mapping
        self.fc_name = fc_name
        self.random_map = random_map
        self.class_agnostic = class_agnostic

    def _get_unis(self, num_uni=None):
        if self.random_map:
            assert num_uni is not None
            uni_cands = range(num_uni)
            num_sep = len(self.label_mapping.labels)
            unis = random.choices(uni_cands, k=num_sep)
        else:
            cls_unis = []
            assert self.label_mapping.has_sep2uni()
            for sep in self.label_mapping.labels:
                cls_unis.append(self.label_mapping.sep2uni(sep))
        cls_unis.append(-1) # append bg

        if self.class_agnostic:
            return cls_unis, None
        else:
            reg_unis = []
            for v in unis:
                reg_v = v*4
                reg_unis.extend([reg_v, reg_v+1, reg_v+2, reg_v+3])
            return cls_unis, reg_unis

    def operate_on(self, state_dict):
        cls_name = self.fc_name
        uni_cls_w = state_dict[f'{cls_name}.weight']
        uni_cls_b = state_dict[f'{cls_name}.bias']
        num_uni_cls = uni_cls_b.shape[0]
        cls_unis, reg_unis = self._get_unis(num_uni_cls)
        sep_cls_w = uni_cls_w[cls_unis, :]
        sep_cls_b = uni_cls_b[cls_unis]
        state_dict[f'{cls_name}.weight'] = sep_cls_w
        state_dict[f'{cls_name}.bias'] = sep_cls_b

        if not self.class_agnostic:
            reg_name = cls_name.replace('cls', 'reg')
            uni_reg_w = state_dict[f'{reg_name}.weight']
            uni_reg_b = state_dict[f'{reg_name}.bias']
            sep_reg_w = uni_reg_w[reg_unis, :]
            sep_reg_b = uni_reg_b[reg_unis]
            state_dict[f'{reg_name}.weight'] = sep_reg_w
            state_dict[f'{reg_name}.bias'] = sep_reg_b

        return state_dict
