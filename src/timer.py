from datetime import datetime


class Timer:
    def __init__(self, name: str):
        if name is None:
            raise Exception("Timer name required")
        
        self.name = name

    def __enter__(self):
        self.start = datetime.now()
        if self.name:
            print(f"{self.name} started")
        return self

    def __exit__(self, *args):
        elapsed = datetime.now() - self.start
        prefix = f"{self.name} finished, "
        print(f"{prefix}time elapsed: {elapsed}")
