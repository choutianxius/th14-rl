from random import choice
from models.base import ModelBase


class RandomWalk(ModelBase):
    def predict(self, *args):
        return choice(range(10)), None
