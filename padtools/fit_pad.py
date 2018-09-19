from abc import ABC
from enum import auto, IntEnum, EnumMeta
from itertools import count
from typing import Dict, Iterable, Set, Callable

from numpy import array, pi, ndarray, stack, fromiter
from numpy.linalg import pinv

from . import solved_helium_eq as he
from . import solved_neon_eq as ne

__all__ = (
    'TargetHeliumPad',
    'TargetNeonPad',
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


class TargetPad(ABC):
    XKEYS: (Iterable[int], EnumMeta)
    YKEYS: (Iterable[int], EnumMeta)
    ZKEYS: EnumMeta = ZKeys
    ETA_REF: IntEnum
    WONLY_XKEYS: Set[IntEnum]
    YMAT: Callable[..., ndarray]
    YJACMAT: Callable[..., ndarray]

    def __init__(self,
                 w2w_beta1_amp: float, w2w_beta1_shift: float, w2w_beta2: float,
                 w2w_beta3_amp: float, w2w_beta3_shift: float, w2w_beta4: float,
                 wonly_beta2: float, wonly_beta4: float,
                 w2w_beta1_amp_err: float = None, w2w_beta1_shift_err: float = None, w2w_beta2_err: float = None,
                 w2w_beta3_amp_err: float = None, w2w_beta3_shift_err: float = None, w2w_beta4_err: float = None,
                 wonly_beta2_err: float = None, wonly_beta4_err: float = None,
                 amp_weight: float = 1, shift_weight: float = 1, even_weight: float = 1,
                 w2w_weight: float = 1, wonly_weight: float = 1,
                 xfixed: Dict[IntEnum, float] = None):
        if xfixed is None:
            xfixed = {self.ETA_REF: 0}
        for key in xfixed:
            if key not in self.XKEYS:
                ValueError('Fixed key {} is unknown!'.format(key))
        self.__xfixed = xfixed
        self.__xkeys_varying = [k for k in self.XKEYS if k not in xfixed]

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
            (w2w_beta1_amp_weight * w2w_weight * amp_weight +
             w2w_beta2_weight * w2w_weight * even_weight +
             w2w_beta3_amp_weight * w2w_weight * amp_weight +
             w2w_beta4_weight * w2w_weight * even_weight),  # w2w_b0_weight
            w2w_beta1_amp_weight * w2w_weight * amp_weight,
            w2w_beta1_shift_weight * w2w_weight * shift_weight,
            w2w_beta2_weight * w2w_weight * even_weight,
            w2w_beta3_amp_weight * w2w_weight * amp_weight,
            w2w_beta3_shift_weight * w2w_weight * shift_weight,
            w2w_beta4_weight * w2w_weight * even_weight,
            wonly_beta2_weight * wonly_weight * even_weight,
            wonly_beta4_weight * wonly_weight * even_weight,
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
        return fromiter(
            (self.xfixed[k] if k in self.xfixed else xargs[next(inx)] for k in self.XKEYS),
            dtype=xargs.dtype,
        )

    @classmethod
    def wonly_xmask(cls, xargs_arranged: ndarray) -> ndarray:
        ret = xargs_arranged.copy()
        where = list(set(cls.XKEYS) - cls.WONLY_XKEYS)
        ret[where] = 0
        return ret

    @classmethod
    def zmat(cls, xargs_arranged: ndarray) -> ndarray:
        w2w = cls.YMAT(*xargs_arranged)[:, 0]
        w = cls.YMAT(*cls.wonly_xmask(xargs_arranged))[:, 0]
        yk = cls.YKEYS
        zk = cls.ZKEYS
        ret = stack([*w2w, *w[[yk.B2, yk.B4]]])
        ret[[zk.W2W_BETA1_AMP, zk.W2W_BETA2, zk.W2W_BETA3_AMP, zk.W2W_BETA4]] = (
                w2w[[yk.B1_AMP, yk.B2, yk.B3_AMP, yk.B4]] / w2w[yk.B0]
        )
        ret[[zk.WONLY_BETA2, zk.WONLY_BETA4]] = (
                w[[yk.B2, yk.B4]] / w[yk.B0]
        )
        return ret

    @classmethod
    def zjacmat(cls, xargs_arranged: ndarray) -> ndarray:
        w2w = cls.YMAT(*xargs_arranged)[:, 0]
        w = cls.YMAT(*cls.wonly_xmask(xargs_arranged))[:, 0]
        w2wjac = cls.YJACMAT(*xargs_arranged)
        wjac = cls.YJACMAT(*cls.wonly_xmask(xargs_arranged))
        yk = cls.YKEYS
        zk = cls.ZKEYS
        ret = stack([*w2wjac, *wjac[[yk.B2, yk.B4]]])
        ret[[zk.W2W_BETA1_AMP, zk.W2W_BETA2, zk.W2W_BETA3_AMP, zk.W2W_BETA4], :] = (
                w2w[[yk.B1_AMP, yk.B2, yk.B3_AMP, yk.B4], None] / w2w[yk.B0]
                * ((w2wjac[[yk.B1_AMP, yk.B2, yk.B3_AMP, yk.B4], :]
                    / w2w[[yk.B1_AMP, yk.B2, yk.B3_AMP, yk.B4], None])
                   - (w2wjac[yk.B0, :] / w2w[yk.B0, None])[None, :])
        )
        ret[[zk.WONLY_BETA2, zk.WONLY_BETA4], :] = (
                w[[yk.B2, yk.B4], None] / w[yk.B0]
                * ((wjac[[yk.B2, yk.B4], :]
                    / w[[yk.B2, yk.B4], None])
                   - (wjac[yk.B0, :] / w[yk.B0, None])[None, :])
        )
        return ret

    @classmethod
    def xjacmat_byz(cls, xargs_arranged: ndarray) -> ndarray:
        called = cls.zjacmat(xargs_arranged)  # shape: (z,x)
        return pinv(called)  # shape: (x,z)

    @staticmethod
    def __norm_phases(zmat_called: ndarray) -> ndarray:
        ret = zmat_called.copy()
        zk = TargetPad.ZKEYS
        ret[[zk.W2W_BETA1_SHIFT, zk.W2W_BETA3_SHIFT]] = (
                (zmat_called[[zk.W2W_BETA1_SHIFT, zk.W2W_BETA3_SHIFT]] + pi) % (2 * pi) - pi
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
        xerror = ((self.xjacmat_byz(xargs_arranged) ** 2) @ (self.zerror ** 2)) ** 0.5
        zmat_called = self.zmat(xargs_arranged)
        zjacmat_called = self.zjacmat(xargs_arranged)

        print("{:18s}{:>12s}{:>12s}{:>12s}{:>12s}{}".format(
            "", "", "", "", "",
            "".join("{:>12s}".format("{}".format(k.name.lower())) for k in self.XKEYS),
        ))
        for k, arr in (("at:", xargs_arranged),
                       ("error:", xerror)):
            print("{:18s}{:>12s}{:>12s}{:>12s}{:>12s}{}".format(
                k, "", "", "", "",
                "".join("{:> 12.3f}".format(a) for a in arr),
            ))
        print()
        print("{:18s}{:>12s}{:>12s}{:>12s}{:>12s}{}".format(
            "", "target", "examined", "diff", "weight",
            "".join("{:>12s}".format("d/d{}".format(k.name.lower())) for k in self.XKEYS),
        ))
        for k, *o, jac in zip(self.ZKEYS,
                              self.zintercept,
                              zmat_called,
                              self.__norm_phases(self.zintercept - zmat_called),
                              self.zweight,
                              zjacmat_called):
            print("{:<18s}{:> 12.3f}{:> 12.3f}{:> 12.3f}{:>12.0f}{}".format(
                "{}:".format(k.name.lower()), *o,
                "".join("{:> 12.3f}".format(j) for j in jac),
            ))


# %%
class TargetHeliumPad(TargetPad):
    XKEYS = he.XKeys
    YKEYS = he.YKeys
    ETA_REF = he.XKeys.ETA_D
    WONLY_XKEYS = {he.XKeys.C_SPS, he.XKeys.C_DPS, he.XKeys.ETA_S, he.XKeys.ETA_D}
    YMAT = he.ymat_lambdified
    YJACMAT = he.yjacmat_lambdified


class TargetNeonPad(TargetPad):
    XKEYS = ne.XKeys
    YKEYS = ne.YKeys
    ETA_REF = ne.XKeys.ETA_F
    WONLY_XKEYS = {ne.XKeys.C_PSP, ne.XKeys.C_PDP, ne.XKeys.C_FDP, ne.XKeys.ETA_PSP, ne.XKeys.ETA_PDP, ne.XKeys.ETA_F}
    YMAT = ne.ymat_lambdified
    YJACMAT = ne.yjacmat_lambdified
