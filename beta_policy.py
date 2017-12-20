def constant_beta(step, total_step):
    return 1.0


def linear_beta(step, total_step):
    return min([1.0, round(float(step)/float(total_step-1) + 0.01, 2)])