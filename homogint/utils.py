def time_step(dt):
    def scale(vf):
        def scaled_vf(x):
            return dt*vf(x)
        return scaled_vf
    return scale
