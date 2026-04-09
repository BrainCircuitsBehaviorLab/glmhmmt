import numpy as np
from numba import njit, prange

@njit
def onset_offset_from_codes(stim, delay, t1, t2, t3, t4):
    # stim: 0 VG,1 SS,2 SM,3 SL,4 SIL ; delay: 0 DS,1 DM,2 DL
    if stim == 0:
        return 0.0, 0.0
    elif stim == 1:  # SS
        if delay == 0:
            return t2, t3
        elif delay == 1:
            return t1, t2
        else:
            return 0.0, t1
    elif stim == 2:  # SM
        if delay == 0:
            return t1, t3
        else:
            return 0.0, t2
    elif stim == 3:  # SL
        return 0.0, t3
    else:            # SIL
        return 0.0, t4


@njit
def S_value(t, amp, d, onset, offset):
    if t < onset:
        return 0.0
    elif t <= offset:
        return amp
    else:
        tail_end = offset + d
        if d > 0.0 and abs(offset - onset) >= 1e-5 and t <= tail_end:
            return amp * (1.0 - (t - offset) / d)
        return 0.0


@njit
def U_spatial_value(t, U_amp, U_base, t1, t2, t3, t4, w1, w2, w3, w4):
    r1 = t * w1
    if r1 < 0.0: r1 = 0.0
    elif r1 > 1.0: r1 = 1.0

    r2 = (t - t1) * w2
    if r2 < 0.0: r2 = 0.0
    elif r2 > 1.0: r2 = 1.0

    r3 = (t - t2) * w3
    if r3 < 0.0: r3 = 0.0
    elif r3 > 1.0: r3 = 1.0

    r4 = (t - t3) * w4
    if r4 < 0.0: r4 = 0.0
    elif r4 > 1.0: r4 = 1.0

    return 0.25 * U_amp * (r1 + r2 + r3 + r4) + U_base


@njit
def U_ext_value(t, amp, onset, t4):
    # en tu Julia CPU: (t < onset) ? 0 : amp   (hasta t4)
    return 0.0 if t < onset else amp

@njit
def drift_numba(x1, x2, IL, IC, IR, sL, sC, sR):
    x1sq = x1 * x1
    x2sq = x2 * x2
    s_sum  = sC + sL
    s_diff = sC - sL

    # F1
    F1 = 5.0 * (IL - IC) + 20.0 * x1 * x2
    F1 -= 1.9047619047619 * x1 * (x1sq + 3.0 * x2sq)
    F1 += 5.0 * x1 * s_sum
    F1 += 10.0 * x1 * (0.904761904761905 * (IC + IL) - 0.0952380952380951 * IR +
                       0.226190476190476 * s_sum - 0.0238095238095238 * sR)
    F1 -= 10.0 * x2 * (IC - IL + 0.25 * s_diff)
    F1 -= 5.0 * (x2 + 0.25) * s_diff

    # F2
    F2 = (3.33333333333333 * (IL - IC) * x1 +
          3.09523809523809  * (IL + IC) * x2 +
          1.66666666666667  * (IL + IC) +
          13.0952380952381  * IR * x2 - 3.33333333333333 * IR)
    F2 += (-1.9047619047619 * x1sq * x2 + 3.33333333333333 * x1sq -
           10.0 * x2sq - 1.9047619047619 * x2 * (x1sq + 3.0 * x2sq))
    F2 += (2.44047619047619 * x2 * s_sum + 9.94047619047619 * x2 * sR +
           0.416666666666667 * s_sum - 0.833333333333333 * sR)

    return F1, F2

@njit
def simulate_trial_choice(stim, delay, side_code,
                          t1, t2, t3, t4,
                          sL, sC, sR,
                          noise_amp, S_amp, dS, U_amp, U_base, U_on, U_ext_amp,
                          dt, th1, th2, th3):
    N = int(t4 / dt)
    if N <= 0:
        return -1

    onset, offset = onset_offset_from_codes(stim, delay, t1, t2, t3, t4)

    # precompute slopes
    w1 = 0.0 if t1 <= 0.0 else 1.0 / t1
    d21 = t2 - t1
    w2 = 0.0 if d21 <= 0.0 else 1.0 / d21
    d32 = t3 - t2
    w3 = 0.0 if d32 <= 0.0 else 1.0 / d32
    d43 = t4 - t3
    w4 = 0.0 if d43 <= 0.0 else 1.0 / d43

    # noise coefs (igual que Julia)
    srt = np.sqrt(dt)
    s1_coef = noise_amp * (srt * 0.5)
    s2_coef = noise_amp * (srt / 6.0)

    x1 = 0.0
    x2 = 0.0

    for k in range(1, N + 1):
        tt = (k - 1) * dt

        Sval = S_value(tt, S_amp, dS, onset, offset)
        Uval = U_spatial_value(tt, U_amp, U_base, t1, t2, t3, t4, w1, w2, w3, w4) + U_ext_value(tt, U_ext_amp, onset, t4)

        # Inputs por lado
        if side_code == 0:      # L correct
            IL = Sval + Uval; IC = Uval;       IR = Uval
        elif side_code == 1:    # C
            IL = Uval;       IC = Sval + Uval; IR = Uval
        else:                   # R (2
            IL = Uval;       IC = Uval;        IR = Sval + Uval if side_code == 2 else Uval

        # ruido (3 gauss)
        z0 = np.random.randn()
        z1 = np.random.randn()
        z2 = np.random.randn()
        n1 = s1_coef * (z0 - z1)
        n2 = s2_coef * (z0 + z1 - 2.0 * z2)

        # Heun predictor/corrector
        f1a, f2a = drift_numba(x1, x2, IL, IC, IR, sL, sC, sR)
        x1p = x1 + f1a * dt + n1
        x2p = x2 + f2a * dt + n2
        f1b, f2b = drift_numba(x1p, x2p, IL, IC, IR, sL, sC, sR)

        x1 = x1 + 0.5 * (f1a + f1b) * dt + n1
        x2 = x2 + 0.5 * (f2a + f2b) * dt + n2

    r1 =  x1 + x2
    r2 = -x1 + x2
    r3 = -2.0 * x2

    if (r1 > r2) and (r1 > r3) and (r1 > th1):
        return 0
    elif (r2 > r1) and (r2 > r3) and (r2 > th2):
        return 1
    elif (r3 > r1) and (r3 > r2) and (r3 > th3):
        return 2
    else:
        return -1
    
@njit(parallel=True)
def get_choices_varying_numba(stimd, delayd, side,
                              t1, t2, t3, t4,
                              theta_trials,  # (Ntr, 10) float32
                              dt, th1, th2, th3):
    Ntr = stimd.shape[0]
    out = np.empty(Ntr, dtype=np.int8)

    for i in prange(Ntr):
        sL = theta_trials[i, 0]
        sC = theta_trials[i, 1]
        sR = theta_trials[i, 2]
        noise_amp = theta_trials[i, 3]
        S_amp = theta_trials[i, 4]
        dS = theta_trials[i, 5]
        U_amp = theta_trials[i, 6]
        U_base = theta_trials[i, 7]
        U_on = theta_trials[i, 8]    # si no lo usas, ignóralo
        U_ext_amp = theta_trials[i, 9]

        out[i] = simulate_trial_choice(
            stimd[i], delayd[i], side[i],
            t1[i], t2[i], t3[i], t4[i],
            sL, sC, sR,
            noise_amp, S_amp, dS, U_amp, U_base, U_on, U_ext_amp,
            dt, th1, th2, th3
        )
    return out