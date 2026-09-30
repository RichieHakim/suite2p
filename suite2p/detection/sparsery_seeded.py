"""
Sparsery with access to its internal state, for seeding ROIs at given locations.

``sparsery_with_state`` runs the same greedy loop as ``sparsedetect.sparsery`` and
returns, besides the usual outputs, the state at the stop: the residual movie
(ROI activity subtracted) and the multiscale maps. Seeded searches can then
continue from where sparsery stopped. It can also forbid peaks whose seed square
would touch given pixels (``blocked``), so no ROI starts on those cells; other
ROIs may still grow over them.

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
                           two_comps, multiscale_mask, extendROI)

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


def _blocked_per_scale(blocked, gxy, lxs):
    """
    Per scale: grid points whose seed square (``add_square`` of size lxs[j] at the
    grid point's pixel position, as the sparsery loop builds it) contains a blocked pixel.
    """
    Ly, Lx = blocked.shape
    csum = np.zeros((Ly + 1, Lx + 1), dtype=np.int64)  ## integral image
    csum[1:, 1:] = blocked.cumsum(axis=0).cumsum(axis=1)
    out = []
    for g, ls in zip(gxy[:-1], lxs):
        lhf = int((ls - 1) / 2)
        y, x = g[1].astype(int), g[0].astype(int)  ## (Lyp, Lxp) pixel position of each grid point
        y0, y1 = np.clip(y - lhf, 0, Ly), np.clip(y - lhf + ls, 0, Ly)
        x0, x1 = np.clip(x - lhf, 0, Lx), np.clip(x - lhf + ls, 0, Lx)
        out.append((csum[y1, x1] - csum[y0, x1] - csum[y1, x0] + csum[y0, x0]) > 0)
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
        Boolean (Lyc, Lxc). A peak is never taken if its seed square (at that
        peak's scale) contains a True pixel, so no ROI starts on those cells.
        ROIs started elsewhere may still grow over them.

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
    allowed = None if blocked is None else [~b for b in _blocked_per_scale(blocked, gxy, lxs)]

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


def make_seed(ypix, xpix, lam, n_pixels, Lyc, Lxc):
    """
    Seed of ``n_pixels`` pixels from a footprint: its brightest pixels, equal weights,
    unit norm, like sparsery's square seeds (3x3, 6x6, 12x12 = 9, 36, 144 pixels).

    If the footprint has fewer pixels, it is padded with rings of neighbors
    (closest to the footprint's weighted center first).

    Parameters
    ----------
    ypix, xpix : numpy.ndarray
        Footprint pixels in binned-movie coordinates (int).
    lam : numpy.ndarray
        Footprint weights (only their order matters).
    n_pixels : int
        Seed size.
    Lyc, Lxc : int
        Binned-movie frame size.

    Returns
    -------
    ypix_seed, xpix_seed : numpy.ndarray
    w_seed : numpy.ndarray
        Equal weights with unit norm.
    """
    order = np.argsort(-lam, kind="stable")
    ys, xs = ypix[order], xpix[order]
    if ys.size < n_pixels:
        yc, xc = np.average(ypix, weights=lam), np.average(xpix, weights=lam)
        ys_ring, xs_ring = ys, xs
        while ys.size < n_pixels:
            ys_ring, xs_ring = extendROI(ys_ring, xs_ring, Lyc, Lxc, 1)
            is_new = ~np.isin(ys_ring * Lxc + xs_ring, ys * Lxc + xs)
            yn, xn = ys_ring[is_new], xs_ring[is_new]
            if yn.size == 0:
                raise ValueError("Footprint cannot be padded to the seed size inside the frame")
            order_new = np.argsort((yn - yc)**2 + (xn - xc)**2, kind="stable")
            ys = np.concatenate([ys, yn[order_new]])
            xs = np.concatenate([xs, xn[order_new]])
    ys, xs = ys[:n_pixels], xs[:n_pixels]
    return ys, xs, np.full(n_pixels, 1.0 / np.sqrt(n_pixels), dtype=np.float32)


def score_trace(tproj, threshold):
    """Sparsery's peak score: sqrt of the summed squares of the bins above threshold."""
    return float(np.sqrt(np.sum(tproj[tproj > threshold]**2)))


def _iter_extend_bounded(ypix, xpix, mov, Lyc, Lxc, active_frames, allowed):
    """iter_extend (sparsedetect) with growth limited to pixels where ``allowed`` is True."""
    npix = 0
    iter = 0
    while npix < 10000:
        npix = ypix.size
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)
        keep = allowed[ypix, xpix]
        ypix, xpix = ypix[keep], xpix[keep]
        usub = mov[np.ix_(active_frames, ypix * Lxc + xpix)]
        lam = usub.mean(axis=0)
        ix = lam > max(0, lam.max() / 5.0)
        if ix.sum() == 0:
            break
        ypix, xpix, lam = ypix[ix], xpix[ix], lam[ix]
        if iter == 0:
            sgn = 1.
        if np.sign(sgn * (ix.sum() - npix)) <= 0:
            break
        else:
            npix = ypix.size
        iter += 1
    lam = lam / np.sum(lam**2)**.5
    return ypix, xpix, lam


def grow_from_footprint(state, ypix, xpix, lam, seed_sizes=(9, 36, 144), radius_allowed=3,
                        subtract=False):
    """
    Seed sparsery's ROI growth from a given footprint on the residual movie.

    Steps, following sparsery's loop: seeds of each size from the footprint's
    brightest pixels -> keep the best-scoring seed -> active bins (projection >
    Th2) -> 3 rounds of mask growth, limited to ``radius_allowed`` pixels around
    the footprint (or unlimited, as in sparsery) -> split check -> final mask,
    trace and scores.

    Parameters
    ----------
    state : dict
        From ``sparsery_with_state``. ``state["mov"]`` is updated if ``subtract``.
    ypix, xpix, lam : numpy.ndarray
        Footprint in binned-movie coordinates (not session coordinates:
        subtract yrange[0], xrange[0]).
    seed_sizes : tuple of int
        Seed sizes in pixels.
    radius_allowed : int or None
        How far (in pixels, 4-connected steps) the mask may grow beyond the footprint.
        ``None``: no limit (sparsery's own ``iter_extend``).
    subtract : bool
        Subtract the ROI's activity from the residual, as sparsery does for an
        accepted ROI.

    Returns
    -------
    out : dict
        "score_seeds": score of each seed size; "n_pixels_seed": best seed size;
        "score_seed": its score (compare with state["threshold_stop"]);
        "n_active": active bins; "ypix", "xpix", "lam": final mask (lam in
        normalized units, unit norm; empty if no active bins); "trace": projection
        of the residual onto the final mask (nbinned,); "score_final": score of that
        trace; "vsplit": split ratio (sparsery splits above 1.25).
    """
    mov, Lyc, Lxc, Th2 = state["mov"], state["Lyc"], state["Lxc"], state["Th2"]

    ## Best seed size
    scores, seeds = [], []
    for n_pixels in seed_sizes:
        ys, xs, w = make_seed(ypix, xpix, lam, n_pixels, Lyc, Lxc)
        scores.append(score_trace(mov[:, ys * Lxc + xs] @ w, Th2))
        seeds.append((ys, xs, w))
    i_best = int(np.argmax(scores))
    ypix0, xpix0, lam0 = seeds[i_best]
    out = {"score_seeds": scores, "n_pixels_seed": int(seed_sizes[i_best]), "score_seed": scores[i_best]}

    ## Allowed growth region: footprint dilated by radius_allowed
    if radius_allowed is not None:
        allowed = np.zeros((Lyc, Lxc), dtype=bool)
        ya, xa = extendROI(ypix, xpix, Lyc, Lxc, radius_allowed)
        allowed[ya, xa] = True
        allowed[ypix0, xpix0] = True

    tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
    active_frames = np.nonzero(tproj > Th2)[0]
    for j in range(3):
        if len(active_frames) < 1:
            break
        if radius_allowed is None:
            ypix0, xpix0, lam0 = iter_extend(ypix0, xpix0, mov, Lyc, Lxc, active_frames)
        else:
            ypix0, xpix0, lam0 = _iter_extend_bounded(ypix0, xpix0, mov, Lyc, Lxc, active_frames, allowed)
        tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
        active_frames = np.nonzero(tproj > Th2)[0]
    if len(active_frames) < 1:
        out.update({"n_active": 0, "ypix": np.zeros(0, int), "xpix": np.zeros(0, int), "lam": np.zeros(0),
                    "trace": tproj, "score_final": 0.0, "vsplit": 0.0})
        return out

    vsplit, ipack = two_comps(mov[:, ypix0 * Lxc + xpix0], lam0, Th2)
    if vsplit > 1.25:
        lam0, xp, active_frames = ipack
        tproj[active_frames] = xp
        ix = lam0 > lam0.max() / 5
        ypix0, xpix0, lam0 = ypix0[ix], xpix0[ix], lam0[ix]
        active_frames = np.nonzero(active_frames)[0] if active_frames.dtype == bool else active_frames

    out.update({"n_active": int(len(active_frames)), "ypix": ypix0.astype(int), "xpix": xpix0.astype(int),
                "lam": lam0, "trace": tproj, "score_final": score_trace(tproj, Th2), "vsplit": float(vsplit)})
    if subtract:
        mov[np.ix_(active_frames, ypix0 * Lxc + xpix0)] -= tproj[active_frames][:, np.newaxis] * lam0
    return out
