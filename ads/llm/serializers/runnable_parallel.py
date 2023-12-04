from langchain.schema.runnable import RunnableParallel
from langchain.load.dump import dumpd
from langchain.load.load import load


class RunnableParallelSerializer:
    @staticmethod
    def type():
        return RunnableParallel.__name__

    @staticmethod
    def load(config: dict, **kwargs):
        steps = config["kwargs"]["steps"]
        steps = {k: load(v, **kwargs) for k, v in steps.items()}
        return RunnableParallel(**steps)

    @staticmethod
    def save(obj):
        serialized = dumpd(obj)
        serialized["_type"] = RunnableParallelSerializer.type()
        return serialized
