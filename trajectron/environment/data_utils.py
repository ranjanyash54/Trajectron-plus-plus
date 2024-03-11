import numpy as np


def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x


'''
This function computes the derivative of the input array x.
If radian is set to True, it first ensures that x is continuous
across π by calling the make_continuous_copy function.
It creates a boolean mask not_nan_mask to filter out NaN
values from the input array x.
It applies this mask to get the non-NaN elements of x.
If there are fewer than 2 non-NaN elements, it returns an array of zeros.
Otherwise, it computes the derivative using np.ediff1d,
which calculates the difference between consecutive elements.
The to_begin parameter is used to specify the value to use as
the difference for the first element (since there's no previous element).
The derivative array is then assigned to dx while maintaining
NaN values for elements corresponding to NaN in the input array.
Finally, it returns the derivative array.
This code essentially provides functions to compute the derivative
of an array while handling special cases such as NaN values and
ensuring continuity across π when working with angles in radians.
'''
def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

    return dx

