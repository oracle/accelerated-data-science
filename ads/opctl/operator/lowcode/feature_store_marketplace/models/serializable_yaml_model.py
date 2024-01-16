from enum import Enum

from typing import Type, Dict


class SerializableYAMLModel:
    yaml_mapping = {}

    def to_dict(self) -> Dict[str, object]:
        d = {}
        for yaml_key, obj_key in self.yaml_mapping.items():
            obj_value = self.__getattribute__(obj_key)
            if obj_value is not None:
                if isinstance(obj_value, SerializableYAMLModel):
                    obj_value = obj_value.to_dict()
                elif isinstance(obj_value, Enum):
                    obj_value = obj_value.value
                d[yaml_key] = obj_value
        return d

    @classmethod
    def from_dict(cls: Type["SerializableYAMLModel"], d: dict):
        instance = cls()
        yaml_mapping = instance.yaml_mapping
        for yaml_key, obj_value in d.items():
            if yaml_key in yaml_mapping:
                instance.__setattr__(yaml_mapping.get(yaml_key), obj_value)
        return instance
