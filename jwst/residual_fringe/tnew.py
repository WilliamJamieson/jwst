import asdf
from .ab import remove_fringe, remove_fringe_scipy, remove_fringe_astropy


def compare():
    afin = asdf.open('rfringe_inputs.asdf')
    tree = afin.tree
    asdf_library = tree['asdf_library']
    history = tree['history']
    bg_fit = tree['bg_fit']
    col = tree['col']
    col_max_amp = tree['col_max_amp']
    col_weight = tree['col_weight']
    col_wnum = tree['col_wnum']
    dffreq = tree['dffreq']
    ffreq = tree['ffreq']
    max_nfringes = tree['max_nfringes']
    min_nfringes = tree['min_nfringes']
    min_snr = tree['min_snr']
    pgram_res = tree['pgram_res']
    pre_contrast = tree['pre_contrast']
    proc_data1 = tree['proc_data'].copy()
    proc_factors1 = tree['proc_factors'].copy()
    proc_data2 = proc_data1.copy()
    proc_factors2 = proc_factors1.copy()
    res_fringe_fit = tree['res_fringe_fit']
    res_fringe_fit_flag = tree['res_fringe_fit_flag']
    res_fringes = tree['res_fringes']
    snr2 = tree['snr2']
    ss = tree['ss']
    weights_feat = tree['weights_feat']
    wpix_num = tree['wpix_num']

    result3 = remove_fringe_astropy(col, 0, ffreq[0], ffreq, dffreq, min_snr, snr2, ss,
                                    min_nfringes, max_nfringes, pgram_res, proc_data2, proc_factors2,
                                    pre_contrast, bg_fit, res_fringes, res_fringe_fit, res_fringe_fit_flag,
                                    wpix_num, col_wnum, col_weight, col_max_amp, weights_feat)
    result2 = remove_fringe_scipy(col, 0, ffreq[0], ffreq, dffreq, min_snr, snr2, ss,
                                  min_nfringes, max_nfringes, pgram_res, proc_data2, proc_factors2,
                                  pre_contrast, bg_fit, res_fringes, res_fringe_fit, res_fringe_fit_flag,
                                  wpix_num, col_wnum, col_weight, col_max_amp, weights_feat)
    result1 = remove_fringe(col, 0, ffreq[0], ffreq, dffreq, min_snr, snr2, ss,
                            min_nfringes, max_nfringes, pgram_res, proc_data1, proc_factors1,
                            pre_contrast, bg_fit, res_fringes, res_fringe_fit, res_fringe_fit_flag,
                            wpix_num, col_wnum, col_weight, col_max_amp, weights_feat)
    return None
