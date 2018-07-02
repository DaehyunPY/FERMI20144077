from enum import auto, IntEnum
from itertools import count
from typing import Dict

from numpy import array, pi, ndarray, stack, fromiter
from numpy.linalg import pinv

from .solved_neon_eq import XKeys, eta_ref, wonly_xkeys, xkeys, YKeys, ymat_lambdified, yjacmat_lambdified

__all__ = (
    'ZKeys',
    'zkeys',
    'TargetPad',
)


# %%
class ZKeys(IntEnum):  # length: 9
    W2W_B0 = 0
    W2W_BETA1_AMP = auto()
    W2W_BETA1_SHIFT = auto()
    W2W_BETA2 = auto()
    W2W_BETA3_AMP = auto()
    W2W_BETA3_SHIFT = auto()
    W2W_BETA4 = auto()
    WONLY_BETA2 = auto()
    WONLY_BETA4 = auto()


zkeys = [k.name.lower() for k in ZKeys]


class TargetPad:
    def __init__(self,
                 w2w_beta1_amp: float, w2w_beta1_shift: float, w2w_beta2: float,
                 w2w_beta3_amp: float, w2w_beta3_shift: float, w2w_beta4: float,
                 wonly_beta2: float, wonly_beta4: float,
                 w2w_beta1_amp_err: float = None, w2w_beta1_shift_err: float = None, w2w_beta2_err: float = None,
                 w2w_beta3_amp_err: float = None, w2w_beta3_shift_err: float = None, w2w_beta4_err: float = None,
                 wonly_beta2_err: float = None, wonly_beta4_err: float = None,
                 amp_weight: float = 1, shift_weight: float = 1, even_weight: float = 1,
                 xfixed: Dict[XKeys, float] = None):
        if xfixed is None:
            xfixed = {eta_ref: 0}
        for key in xfixed:
            if key not in XKeys:
                ValueError('Fixed key {} is unknown!'.format(key))
        self.__xfixed = xfixed
        self.__xkeys_varying = [k for k in XKeys if k not in xfixed]

        if any((w2w_beta1_amp_err is None, w2w_beta1_shift_err is None, w2w_beta2_err is None,
                w2w_beta3_amp_err is None, w2w_beta3_shift_err is None, w2w_beta4_err is None,
                wonly_beta2_err is None, wonly_beta4_err is None)):
            if not all((w2w_beta1_amp_err is None, w2w_beta1_shift_err is None, w2w_beta2_err is None,
                        w2w_beta3_amp_err is None, w2w_beta3_shift_err is None, w2w_beta4_err is None,
                        wonly_beta2_err is None, wonly_beta4_err is None)):
                print('Some *_err keyword arguments are passed but will be ignored!')
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
        self.__zerror = array((
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
        self.__zweight = array((
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
        ))
        self.__zintercept = array((
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
    def xfixed(self):
        return self.__xfixed

    @property
    def zerror(self):
        return self.__zerror

    @property
    def zweight(self):
        return self.__zweight

    @property
    def zintercept(self):
        return self.__zintercept

    def __arrange_xargs(self, xargs: ndarray) -> ndarray:
        inx = count()
        return fromiter((self.xfixed[k] if k in self.xfixed else xargs[next(inx)] for k in XKeys), dtype='double')

    @staticmethod
    def wonly_xmask(xargs_arranged: ndarray) -> ndarray:
        ret = xargs_arranged.copy()
        where = list(set(XKeys) - wonly_xkeys)
        ret[where] = 0
        return ret

    @staticmethod
    def zmat(xargs_arranged: ndarray) -> ndarray:
        w2w = ymat_lambdified(*xargs_arranged)[:, 0]
        wonly = ymat_lambdified(*TargetPad.wonly_xmask(xargs_arranged))[:, 0]
        ret = stack([*w2w, *wonly[[YKeys.B2, YKeys.B4]]])
        ret[[ZKeys.W2W_BETA1_AMP, ZKeys.W2W_BETA2, ZKeys.W2W_BETA3_AMP, ZKeys.W2W_BETA4]] = (
                w2w[[YKeys.B1_AMP, YKeys.B2, YKeys.B3_AMP, YKeys.B4]] / w2w[YKeys.B0]
        )
        ret[[ZKeys.WONLY_BETA2, ZKeys.WONLY_BETA4]] = (
                wonly[[YKeys.B2, YKeys.B4]] / wonly[YKeys.B0]
        )
        return ret

    @staticmethod
    def zjacmat(xargs_arranged: ndarray) -> ndarray:
        w2w = ymat_lambdified(*xargs_arranged)[:, 0]
        w = ymat_lambdified(*TargetPad.wonly_xmask(xargs_arranged))[:, 0]
        w2wjac = yjacmat_lambdified(*xargs_arranged)
        wjac = yjacmat_lambdified(*TargetPad.wonly_xmask(xargs_arranged))
        ret = stack([*w2wjac, *wjac[[YKeys.B2, YKeys.B4]]])
        ret[[ZKeys.W2W_BETA1_AMP, ZKeys.W2W_BETA2, ZKeys.W2W_BETA3_AMP, ZKeys.W2W_BETA4], :] = (
                w2w[[YKeys.B1_AMP, YKeys.B2, YKeys.B3_AMP, YKeys.B4], None] / w2w[YKeys.B0]
                * ((w2wjac[[YKeys.B1_AMP, YKeys.B2, YKeys.B3_AMP, YKeys.B4], :]
                    / w2w[[YKeys.B1_AMP, YKeys.B2, YKeys.B3_AMP, YKeys.B4], None])
                   - (w2wjac[YKeys.B0, :] / w2w[YKeys.B0, None])[None, :])
        )
        ret[[ZKeys.WONLY_BETA2, ZKeys.WONLY_BETA4], :] = (
                w[[YKeys.B2, YKeys.B4], None] / w[YKeys.B0]
                * ((wjac[[YKeys.B2, YKeys.B4], :]
                    / w[[YKeys.B2, YKeys.B4], None])
                   - (wjac[YKeys.B0, :] / w[YKeys.B0, None])[None, :])
        )
        return ret

    @staticmethod
    def xjacmat_byz(xargs_arranged: ndarray) -> ndarray:
        called = TargetPad.zjacmat(xargs_arranged)  # shape: (z,x)
        return pinv(called)  # shape: (x,z)

    @staticmethod
    def __norm_phases(zmat_called: ndarray) -> ndarray:
        ret = zmat_called.copy()
        ret[[ZKeys.W2W_BETA1_SHIFT, ZKeys.W2W_BETA3_SHIFT]] = (
                (zmat_called[[ZKeys.W2W_BETA1_SHIFT, ZKeys.W2W_BETA3_SHIFT]] + pi) % (2 * pi) - pi
        )
        return ret

    def zdiffmat(self, xargs: ndarray) -> ndarray:
        arranged = self.__arrange_xargs(xargs)
        called = self.zmat(arranged)
        return self.zweight ** 0.5 * self.__norm_phases(self.zintercept - called)

    def zdiffjacmat(self, xargs: ndarray) -> ndarray:
        arranged = self.__arrange_xargs(xargs)
        called = self.zjacmat(arranged)
        return self.zweight[:, None] ** 0.5 * -called[:, self.__xkeys_varying]

    def report(self, xargs: ndarray) -> None:
        xargs_arranged = self.__arrange_xargs(xargs)
        zmat_called = self.zmat(xargs_arranged)
        xerror = ((self.xjacmat_byz(xargs_arranged) ** 2) @ (self.zerror ** 2)) ** 0.5

        print("{:<18s} {:>9s} {:>9s}".format("", "value", "error"))
        for k, *o in zip(xkeys, xargs_arranged, xerror):
            print("{:<18s} {:> 9.3f} {:> 9.3f}".format("{}:".format(k), *o))

        print()
        print("{:<18s} {:>9s} {:>9s} {:>9s} {:>9s}".format("", "target", "examined", "diff", "weight"))
        for k, *o in zip(zkeys,
                         self.zintercept,
                         zmat_called,
                         self.__norm_phases(self.zintercept - zmat_called),
                         self.zweight):
            print("{:<18s} {:> 9.3f} {:> 9.3f} {:> 9.3f} {:>9.0f}".format("{}:".format(k), *o))
