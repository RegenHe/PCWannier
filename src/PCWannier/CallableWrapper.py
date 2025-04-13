import cloudpickle

class CallableWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getstate__(self):
        return cloudpickle.dumps(self.func)

    def __setstate__(self, state):
        self.func = cloudpickle.loads(state)
