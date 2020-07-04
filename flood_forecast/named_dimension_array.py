import numpy as np
import typing


class NamedDimensionArray(np.ndarray):
    def __new__(cls, input_array, dimension_names: typing.List[str] = None):
        obj = np.asarray(input_array).view(cls)
        dims = obj.shape
        if dimension_names is None:
            dimension_names = list(range(len(dims)))
        obj.dimension_names = dimension_names
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dimension_names = getattr(obj, 'dimension_names', None)

    def iterate_over_axis(self, axis_name: str) -> typing.Generator:
        assert axis_name in self.dimension_names
        axis_i = self.dimension_names.index(axis_name)
        remaining_dimension_names = [
            dimension_name for dimension_name in self.dimension_names if dimension_name != axis_name]
        sub_arrays = np.rollaxis(self, axis_i)
        for sub_array in sub_arrays:
            yield NamedDimensionArray(sub_array, remaining_dimension_names)

    def apply_along_axis(self, func1d: typing.Any, axis_name: str):
        assert axis_name in self.dimension_names
        axis_i = self.dimension_names.index(axis_name)
        remaining_dimension_names = [
            dimension_name for dimension_name in self.dimension_names if dimension_name != axis_name]
        sub_arrays = np.apply_along_axis(func1d, axis_i, self)
        return NamedDimensionArray(sub_arrays, remaining_dimension_names)
