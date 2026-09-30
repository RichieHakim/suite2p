"""
Sparsery with access to its internal state, for seeding ROIs at given locations.

``sparsery_with_state`` runs the same greedy loop as ``sparsedetect.sparsery`` and
returns, besides the usual outputs, the state at the stop: the residual movie
(ROI activity subtracted) and the multiscale maps. Seeded searches can then
continue from where sparsery stopped. It can also forbid peaks in given pixels
(``blocked``), which leaves those cells' activity in the residual.

With ``blocked=None`` the outputs equal ``sparsedetect.sparsery``; the loop body is
copied from it (suite2p 90be895) and must stay in step with it.
"""
from copy import deepcopy
import logging
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline

from . import utils
from .sparsedetect import (neuropil_subtraction, square_convolution_2d, downsample,
                           threshold_reduce, find_best_scale, add_square, iter_extend,
                           two_comps, multiscale_mask)

logger = logging.getLogger(__name__)


def preprocess_binned_movie(mov_binned, highpass_time):
    """
    Apply detection_wrapper's preprocessing for sparsery to a saved binned movie.

    Parameters
    ----------
    mov_binned : numpy.ndarray
        Binned movie (nbinned, Lyc, Lxc), as saved by the "save_binned_movie"
        setting. Not modified.
    highpass_time : int
        settings["detection"]["highpass_time"].

    Returns
    -------
    mov : numpy.ndarray
        Temporally high-pass filtered movie, float32, same shape.
    sdmov : numpy.ndarray
        Per-pixel standard deviation of frame differences, shape (Lyc, Lxc).
    """
    mov = utils.temporal_high_pass_filter(mov=np.array(mov_binned, dtype=np.float32), width=int(highpass_time))
    sdmov = utils.standard_deviation_over_time(mov, batch_size=1000)
    return mov, sdmov


def _blocked_per_scale(blocked, gxy):
    """Map a pixel mask (Lyc, Lxc) onto each scale's grid via the grid's pixel coordinates."""
    out = []
    for g in gxy[:-1]:
        yy = np.clip(g[1].astype(int), 0, blocked.shape[0] - 1)
        xx = np.clip(g[0].astype(int), 0, blocked.shape[1] - 1)
        out.append(blocked[yy, xx])
    return out


def sparsery_with_state(mov, sdmov, highpass_neuropil, spatial_scale, threshold_scaling,
                        max_ROIs, active_percentile=0, blocked=None):
    """
    Run sparsery and return its state at the stop, optionally forbidding peaks.

    Parameters
    ----------
    mov, sdmov, highpass_neuropil, spatial_scale, threshold_scaling, max_ROIs, active_percentile
        As in ``sparsedetect.sparsery``. ``mov`` is overwritten (as there).
    blocked : numpy.ndarray, optional
        Boolean (Lyc, Lxc). Peaks whose position falls on a True pixel are never
        taken, at any scale, so the activity there stays in the residual.

    Returns
    -------
    new_settings : dict
        As ``sparsedetect.sparsery``.
    stats : list of dict
        As ``sparsedetect.sparsery``.
    state : dict
        "mov": residual movie (nbinned, Lyc * Lxc) in sparsery's normalized units;
        "movu": residual multiscale movies; "V1": score maps at the stop;
        "Lyc", "Lxc", "Lyp", "Lxp", "gxy", "Th2", "vmultiplier", "threshold_stop"
        (= vmultiplier * Th2), "sdmov", "lxs".
    """
    mov = neuropil_subtraction(
        mov=mov / sdmov,
        filter_size=highpass_neuropil)  # subtract low-pass filtered movie

    _, Lyc, Lxc = mov.shape
    LL = np.meshgrid(np.arange(Lxc), np.arange(Lyc))
    gxy = [np.array(LL).astype("float32")]
    dmov = mov
    movu = []

    # downsample movie at various spatial scales
    Lyp, Lxp = np.zeros(5, "int32"), np.zeros(5, "int32")  # downsampled sizes
    for j in range(5):
        movu0 = square_convolution_2d(dmov, 3)
        dmov = 2 * downsample(dmov)
        gxy0 = downsample(gxy[j], False)
        gxy.append(gxy0)
        _, Lyp[j], Lxp[j] = movu0.shape
        movu.append(movu0)

    # spline over scales
    I = np.zeros((len(gxy), gxy[0].shape[1], gxy[0].shape[2]))
    for movu0, gxy0, I0 in zip(movu, gxy, I):
        gmodel = RectBivariateSpline(gxy0[1, :, 0], gxy0[0, 0, :], movu0.max(axis=0),
                                     kx=min(3, gxy0.shape[1] - 1),
                                     ky=min(3, gxy0.shape[2] - 1))
        I0[:] = gmodel(gxy[0][1, :, 0], gxy[0][0, 0, :])
    v_corr = I.max(axis=0)

    scale, estimate_mode = find_best_scale(I=I, spatial_scale=spatial_scale)
    spatscale_pix = 3 * 2**scale
    if isinstance(spatscale_pix, np.ndarray):
        spatscale_pix = spatscale_pix.item()
    Th2 = threshold_scaling * 5 * max(
        1, scale)  # threshold for accepted peaks (scale it by spatial scale)
    vmultiplier = max(1, mov.shape[0] / 1200)
    logger.info("NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f " %
          (estimate_mode.value, spatscale_pix, vmultiplier, vmultiplier * Th2))

    # get standard deviation for pixels for all values > Th2
    v_map = [threshold_reduce(movu0, Th2) for movu0 in movu]
    movu = [movu0.reshape(movu0.shape[0], -1) for movu0 in movu]

    mov = np.reshape(mov, (-1, Lyc * Lxc))
    lxs = 3 * 2**np.arange(5)
    nscales = len(lxs)

    v_max = np.zeros(max_ROIs)
    ihop = np.zeros(max_ROIs)
    v_split = np.zeros(max_ROIs)
    V1 = deepcopy(v_map)
    stats = []
    allowed = None if blocked is None else [~b for b in _blocked_per_scale(blocked, gxy)]

    t0 = time.time()
    for tj in range(max_ROIs):
        # find peaks in stddev"s (never on blocked positions)
        V1_search = V1 if allowed is None else [V1[j] * allowed[j] for j in range(5)]
        v0max = np.array([V1_search[j].max() for j in range(5)])
        imap = np.argmax(v0max)
        imax = np.argmax(V1_search[imap])
        yi, xi = np.unravel_index(imax, (Lyp[imap], Lxp[imap]))
        # position of peak
        yi, xi = gxy[imap][1, yi, xi], gxy[imap][0, yi, xi]
        med = [int(yi), int(xi)]

        # check if peak is larger than threshold * max(1,nbinned/1200)
        v_max[tj] = v0max.max()
        if v_max[tj] < vmultiplier * Th2:
            break
        ls = lxs[imap]

        ihop[tj] = imap

        # make square of initial pixels based on spatial scale of peak
        yi, xi = int(yi), int(xi)
        ypix0, xpix0, lam0 = add_square(yi, xi, ls, Lyc, Lxc)

        # project movie into square to get time series
        tproj = (mov[:, ypix0 * Lxc + xpix0] * lam0[0]).sum(axis=-1)
        if active_percentile > 0:
            threshold = min(Th2, np.percentile(tproj, active_percentile))
        else:
            threshold = Th2
        active_frames = np.nonzero(tproj > threshold)[0]  # frames with activity > Th2

        # extend mask based on activity similarity
        for j in range(3):
            ypix0, xpix0, lam0 = iter_extend(ypix0, xpix0, mov, Lyc, Lxc, active_frames)
            tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
            active_frames = np.nonzero(tproj > threshold)[0]
            if len(active_frames) < 1:
                break
        if len(active_frames) < 1:
            break

        # check if ROI should be split
        v_split[tj], ipack = two_comps(mov[:, ypix0 * Lxc + xpix0], lam0, threshold)
        if v_split[tj] > 1.25:
            lam0, xp, active_frames = ipack
            tproj[active_frames] = xp
            ix = lam0 > lam0.max() / 5
            xpix0 = xpix0[ix]
            ypix0 = ypix0[ix]
            lam0 = lam0[ix]
            ymed = np.median(ypix0)
            xmed = np.median(xpix0)
            imin = np.argmin((xpix0 - xmed)**2 + (ypix0 - ymed)**2)
            med = [ypix0[imin], xpix0[imin]]

        # update residual on raw movie
        mov[np.ix_(active_frames,
                   ypix0 * Lxc + xpix0)] -= tproj[active_frames][:, np.newaxis] * lam0
        # update filtered movie
        ys, xs, lms = multiscale_mask(ypix0, xpix0, lam0, Lyp, Lxp)
        for j in range(nscales):
            movu[j][np.ix_(active_frames, xs[j] + Lxp[j] * ys[j])] -= np.outer(
                tproj[active_frames], lms[j])
            Mx = movu[j][:, xs[j] + Lxp[j] * ys[j]]
            V1[j][ys[j], xs[j]] = (Mx**2 * np.float32(Mx > threshold)).sum(axis=0)**.5

        stats.append({
            "ypix": ypix0.astype(int),
            "xpix": xpix0.astype(int),
            "lam": lam0 * sdmov[ypix0, xpix0],
            "med": med,
            "footprint": ihop[tj]
        })

        if tj % 500 == 0:
            t1 = time.time() - t0
            logger.info(f"ROIs: {tj},\t last score: {v_max[tj]:0.4f}, \t time: {t1:0.2f}sec")

    new_settings = {
        "Vmax": v_max,
        "ihop": ihop,
        "Vsplit": v_split,
        "Vcorr": v_corr,
        "Vmap": np.asanyarray(v_map, dtype="object"),
        "spatscale_pix": spatscale_pix,
    }
    state = {
        "mov": mov, "movu": movu, "V1": V1, "Lyc": Lyc, "Lxc": Lxc, "Lyp": Lyp, "Lxp": Lxp,
        "gxy": gxy, "Th2": Th2, "vmultiplier": vmultiplier,
        "threshold_stop": vmultiplier * Th2, "sdmov": sdmov, "lxs": lxs,
    }
    return new_settings, stats, state
