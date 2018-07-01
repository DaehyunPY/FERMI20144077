from itertools import count, chain
from textwrap import dedent

from numpy import array, pi, ndarray
from numpy.linalg import pinv

from .solved_neon_eq import neon_pad

__all__ = (
    'TargetNeonPad',
)


# %%
class TargetNeonPad:
    __xkeys = ('c_sp', 'c_psp', 'c_pdp', 'c_dp', 'c_fdp', 'eta_s', 'eta_p', 'eta_d', 'eta_f')  # order sensitive!
    __wonly_xwhere = array((False, True, True, False, True, False, False, False, False), dtype='b')

    def __init__(self,
                 w2w_beta1_amp: float, w2w_beta1_shift: float, w2w_beta2: float,
                 w2w_beta3_amp: float, w2w_beta3_shift: float, w2w_beta4: float,
                 wonly_beta2: float, wonly_beta4: float,
                 w2w_beta1_amp_err: float = None, w2w_beta1_shift_err: float = None, w2w_beta2_err: float = None,
                 w2w_beta3_amp_err: float = None, w2w_beta3_shift_err: float = None, w2w_beta4_err: float = None,
                 wonly_beta2_err: float = None, wonly_beta4_err: float = None,
                 amp_weight: float = 1, shift_weight: float = 1, even_weight: float = 1,
                 fixed: dict = None,
                 ):
        if fixed is None:
            fixed = {'eta_f': 0}
        for key in fixed:
            if key not in set(self.xkeys):
                ValueError('Fixed key {} is unknown!'.format(key))
        self.__xfixed = fixed

        if any((w2w_beta1_amp_err is None, w2w_beta1_shift_err is None, w2w_beta2_err is None,
                w2w_beta3_amp_err is None, w2w_beta3_shift_err is None, w2w_beta4_err is None,
                wonly_beta2_err is None, wonly_beta4_err is None)):
            print('Ignoring error analysis!')
            w2w_beta1_amp_err = w2w_beta1_shift_err = w2w_beta2_err = 0
            w2w_beta3_amp_err = w2w_beta3_shift_err = w2w_beta4_err = 0
            wonly_beta2_err = wonly_beta4_err = 0
            w2w_beta1_amp_weight = w2w_beta1_shift_weight = w2w_beta2_weight = 1
            w2w_beta3_amp_weight = w2w_beta3_shift_weight = w2w_beta4_weight = 1
            wonly_beta2_weight = wonly_beta4_weight = 1
        else:
            w2w_beta1_amp_weight = 1 / w2w_beta1_amp_err ** 2
            w2w_beta1_shift_weight = 1 / w2w_beta1_shift_err ** 2
            w2w_beta2_weight = 1 / w2w_beta2_err ** 2
            w2w_beta3_amp_weight = 1 / w2w_beta3_amp_err ** 2
            w2w_beta3_shift_weight = 1 / w2w_beta3_shift_err ** 2
            w2w_beta4_weight = 1 / w2w_beta4_err ** 2
            wonly_beta2_weight = 1 / wonly_beta2_err ** 2
            wonly_beta4_weight = 1 / wonly_beta4_err ** 2
        self.__yerror = array((
            0,  # w2w_b0_err
            w2w_beta1_amp_err,
            w2w_beta1_shift_err,
            w2w_beta2_err,
            w2w_beta3_amp_err,
            w2w_beta3_shift_err,
            w2w_beta4_err,
            wonly_beta2_err,
            wonly_beta4_err,
        ))
        self.__yweight = array((
            (w2w_beta1_amp_weight * amp_weight +
             w2w_beta2_weight * even_weight +
             w2w_beta3_amp_weight * amp_weight +
             w2w_beta4_weight * even_weight),  # w2w_b0_weight
            w2w_beta1_amp_weight * amp_weight,
            w2w_beta1_shift_weight * shift_weight,
            w2w_beta2_weight * even_weight,
            w2w_beta3_amp_weight * amp_weight,
            w2w_beta3_shift_weight * shift_weight,
            w2w_beta4_weight * even_weight,
            wonly_beta2_weight * even_weight,
            wonly_beta4_weight * even_weight,
        )) ** 0.5
        self.__yintercept = array((
            1,  # w2w_b0
            w2w_beta1_amp,
            w2w_beta1_shift,
            w2w_beta2,
            w2w_beta3_amp,
            w2w_beta3_shift,
            w2w_beta4,
            wonly_beta2,
            wonly_beta4,
        ))

    @property
    def xkeys(self):
        return self.__xkeys

    @property
    def wonly_xwhere(self):
        return self.__wonly_xwhere

    @property
    def xfixed(self):
        return self.__xfixed

    @property
    def yerror(self):
        return self.__yerror

    @property
    def yweight(self):
        return self.__yweight

    @property
    def yintercept(self):
        return self.__yintercept

    def arrange_xargs(self, xargs: ndarray) -> list:
        inx = count()
        return [self.xfixed[k] if k in self.xfixed else xargs[next(inx)] for k in set(self.xkeys)]

    @staticmethod
    def phase_remainder(y: ndarray) -> ndarray:
        r = y.copy()
        where = array((False, False, True, False, False, True, False, False, False), dtype='b')
        r[where] = (r[where] + pi) % (2 * pi) - pi
        return r

    def ydiff(self, xargs: ndarray) -> ndarray:
        arranged = self.arrange_xargs(xargs)
        w2w = neon_pad(*arranged)['summed']
        wonly = neon_pad(*(arg if b else 0 for arg, b in zip(arranged, self.wonly_xwhere)))['summed']
        fx = array((
            w2w['b0'],
            w2w['b1_amp'] / w2w['b0'],
            w2w['b1_shift'],
            w2w['b2'] / w2w['b0'],
            w2w['b3_amp'] / w2w['b0'],
            w2w['b3_shift'],
            w2w['b4'] / w2w['b0'],
            wonly['b2'] / wonly['b0'],
            wonly['b4'] / wonly['b0'],
        ))
        return self.yweight * self.phase_remainder(self.yintercept - fx)

    def xerror(self, ydiff_jac: ndarray) -> ndarray:
        return (pinv(ydiff_jac / self.yweight[:, None]) ** 2) @ (self.yerror ** 2)

    def report(self, xargs: ndarray) -> None:
        arranged = self.arrange_xargs(xargs)
        w2w = neon_pad(*arranged)['summed']
        wonly = neon_pad(*(arg if b else 0 for arg, b in zip(arranged, self.wonly_xwhere)))['summed']
        fx = array((
            w2w['b0'],
            w2w['b1_amp'] / w2w['b0'],
            w2w['b1_shift'],
            w2w['b2'] / w2w['b0'],
            w2w['b3_amp'] / w2w['b0'],
            w2w['b3_shift'],
            w2w['b4'] / w2w['b0'],
            wonly['b2'] / wonly['b0'],
            wonly['b4'] / wonly['b0'],
        ))
        print(dedent("""\
                          target  examined  diff  weight
        w2w_b0:          {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta1_amp:   {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta1_shift: {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta2:       {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta3_amp:   {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta3_shift: {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        w2w_beta4:       {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        wonly_beta2:     {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}
        wonly_beta4:     {: 6.3f}  {: 6.3f}    {: 6.3f} {:.0f}""".format(
            *chain.from_iterable(zip(self.yintercept,
                                     fx,
                                     self.phase_remainder(self.yintercept - fx),
                                     self.yweight,
                                     )))))
