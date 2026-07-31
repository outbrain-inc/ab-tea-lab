import inspect
from typing import Any, Self


class ParamsMixin:
    """Mixin providing get_params / set_params via __init__ introspection."""

    def get_params(self) -> dict[str, Any]:
        sig = inspect.signature(self.__class__.__init__)
        return {
            name: getattr(self, name)
            for name, param in sig.parameters.items()
            if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }

    def set_params(self, **params: object) -> Self:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")
            setattr(self, key, value)
        return self
