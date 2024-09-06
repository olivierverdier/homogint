from padexp import Exponential # type: ignore

Exp = Exponential(order=16)

def exponential(xi):
    """
    Maps a Lie algebra vector to a group element.
    """
    return Exp(xi)[0]
