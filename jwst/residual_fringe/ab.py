from functools import partial
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits

from . import utils
from . import nutils
# import astropy.units as u

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def remove_fringe(col, fn, ff, ffreq, dffreq, min_snr, snr2, ss,
                  min_nfringes, max_nfringes, pgram_res, proc_data, proc_factors,
                  pre_contrast, bg_fit, res_fringes, res_fringe_fit, res_fringe_fit_flag,
                  wpix_num, col_wnum, col_weight, col_max_amp, weights_feat):
    if ff > 1e-03:
        log.debug(
            ' starting ffreq = {}'.format(ff))

        # check if snr criteria is met for fringe component, should always be true for fringe 1
        if snr2 > min_snr[fn]:
            log.debug(
                " fitting spectral baseline")
            bg_fit, bgindx, fitter = \
                utils.fit_1d_background_complex(proc_data, weights_feat,
                                                col_wnum, ffreq=ffreq[fn])
    return
    #         # get the residual fringes as fraction of signal
    #         res_fringes = np.divide(proc_data, bg_fit, out=np.zeros_like(proc_data),
    #                                 where=bg_fit != 0)
    #         res_fringes = np.subtract(
    #             res_fringes, 1, where=res_fringes != 0)
    #         res_fringes *= np.where(
    #             col_weight > 1e-07, 1, 1e-08)
    #         # get the pre-correction contrast using fringe component 1
    #         if fn == 0:
    #             pre_contrast, quality = utils.fit_quality(col_wnum,
    #                                                       res_fringes,
    #                                                       weights_feat,
    #                                                       ffreq[0],
    #                                                       dffreq[0])

    #             log.debug(
    #                 " pre-correction contrast = {}".format(pre_contrast))

    #         # fit the residual fringes
    #         log.debug(" set up bayes ")
    #         res_fringe_fit, wpix_num, opt_nfringe, peak_freq, freq_min, freq_max = \
    #             utils.fit_1d_fringes_bayes_evidence(res_fringes,
    #                                                 weights_feat,
    #                                                 col_wnum,
    #                                                 ffreq[fn],
    #                                                 dffreq[fn],
    #                                                 min_nfringes=min_nfringes[fn],
    #                                                 max_nfringes=max_nfringes[fn],
    #                                                 pgram_res=pgram_res[fn])

    #         # check for fit blowing up, reset rfc fit to 0, raise a flag
    #         log.debug(
    #             "check residual fringe fit for bad fit regions")
    #         res_fringe_fit, res_fringe_fit_flag = utils.check_res_fringes(res_fringe_fit,
    #                                                                       col_max_amp)

    #         # correct for residual fringes
    #         log.debug(
    #             " divide out residual fringe fit, get fringe corrected column")
    #         _, _, _, env, u_x, u_y = utils.fit_envelope(np.arange(res_fringe_fit.shape[0]),
    #                                                     res_fringe_fit)
    #         rfc_factors = 1 / \
    #             (res_fringe_fit * (col_weight >
    #                                1e-05).astype(int) + 1)
    #         proc_data *= rfc_factors
    #         proc_factors *= rfc_factors

    #         # handle nans or infs that may exist
    #         proc_data[proc_data == np.inf] = 0
    #         proc_data = np.nan_to_num(
    #             proc_data)
    #     return opt_nfringe, peak_freq, freq_min, freq_max, bgindx, proc_data, proc_factors
    # return None


def remove_fringe_astropy(col, fn, ff, ffreq, dffreq, min_snr, snr2, ss,
                          min_nfringes, max_nfringes, pgram_res, proc_data, proc_factors,
                          pre_contrast, bg_fit, res_fringes, res_fringe_fit, res_fringe_fit_flag,
                          wpix_num, col_wnum, col_weight, col_max_amp, weights_feat):
    if ff > 1e-03:
        log.debug(
            ' starting ffreq = {}'.format(ff))

        # check if snr criteria is met for fringe component, should always be true for fringe 1
        if snr2 > min_snr[fn]:
            log.debug(
                " fitting spectral baseline")
            bg_fit, bgindx, fitter = \
                nutils.fit_1d_background_complex(proc_data, weights_feat,
                                                 col_wnum, ffreq=ffreq[fn])
            return
            # get the residual fringes as fraction of signal
            res_fringes = np.divide(proc_data, bg_fit, out=np.zeros_like(proc_data),
                                    where=bg_fit != 0)
            res_fringes = np.subtract(
                res_fringes, 1, where=res_fringes != 0)
            res_fringes *= np.where(
                col_weight > 1e-07, 1, 1e-08)
            # get the pre-correction contrast using fringe component 1
            if fn == 0:
                pre_contrast, quality = utils.fit_quality(col_wnum,
                                                          res_fringes,
                                                          weights_feat,
                                                          ffreq[0],
                                                          dffreq[0])

                log.debug(
                    " pre-correction contrast = {}".format(pre_contrast))

            # fit the residual fringes
            log.debug(" set up bayes ")
            res_fringe_fit, wpix_num, opt_nfringe, peak_freq, freq_min, freq_max = \
                nutils.fit_1d_fringes_bayes_evidence(res_fringes,
                                                     weights_feat,
                                                     col_wnum,
                                                     ffreq[fn],
                                                     dffreq[fn],
                                                     min_nfringes=min_nfringes[fn],
                                                     max_nfringes=max_nfringes[fn],
                                                     pgram_res=pgram_res[fn])

            # check for fit blowing up, reset rfc fit to 0, raise a flag
            log.debug(
                "check residual fringe fit for bad fit regions")
            res_fringe_fit, res_fringe_fit_flag = utils.check_res_fringes(res_fringe_fit,
                                                                          col_max_amp)

            # correct for residual fringes
            log.debug(
                " divide out residual fringe fit, get fringe corrected column")
            _, _, _, env, u_x, u_y = utils.fit_envelope(np.arange(res_fringe_fit.shape[0]),
                                                        res_fringe_fit)
            rfc_factors = 1 / \
                (res_fringe_fit * (col_weight >
                                   1e-05).astype(int) + 1)
            proc_data *= rfc_factors
            proc_factors *= rfc_factors

            # handle nans or infs that may exist
            proc_data[proc_data == np.inf] = 0
            proc_data = np.nan_to_num(
                proc_data)
        return opt_nfringe, peak_freq, freq_min, freq_max, bgindx, proc_data, proc_factors
    return None
