from padexp import Exponential # type: ignore

Exp = Exponential(order=16)

def exponential(xi):
    return Exp(xi)[0]
