from random import choice


class RandomWalk:
    def predict(self, *args):
        return choice(range(10)), None
