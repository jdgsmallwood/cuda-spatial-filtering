import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from phase_structure_simulation import (
        COARSE_HZ,
        FINE_HZ,
        PN_OFFSETS_HZ,
        fitted_slopes,
        phase_terms,
        pn_tooth_phase,
    )

    return (
        COARSE_HZ,
        FINE_HZ,
        PN_OFFSETS_HZ,
        fitted_slopes,
        mo,
        np,
        phase_terms,
        plt,
        pn_tooth_phase,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Why do phase slopes differ within and between coarse channels?

    This is a **relative phase** simulation of two receiver streams. It is not
    a model of the Lambda FPGA circuit. Select one effect at a time, then use
    **Custom sliders** to combine them. The plots show eight adjacent coarse
    channels, 27 retained fine bins in each, and four illustrative PN comb teeth.

    A delay before channelization rotates phase with **absolute RF frequency**.
    It must give the same slope within a coarse channel and between channel
    centres. The two other controls deliberately act in different frequency
    coordinates:

    \[
    \phi(c,\delta f) =
       2\pi(f_c-f_0+\delta f)\tau_{\rm input}
       + 2\pi(f_c-f_0)\tau_{\rm coarse}
       + 2\pi\delta f\tau_{\rm post}
       + \phi_{\rm PN}(c,\delta f).
    \]

    Here \(f_c\) is the coarse centre and \(\delta f\) is frequency relative
    to that centre. The PN term appears only on the injected test comb, not on
    sky data. The sign convention is arbitrary; conjugating all phases changes
    every sign but leaves the comparison intact.
    """)
    return


@app.cell
def _(mo):
    scenario = mo.ui.dropdown(
        options=[
            "Pure input delay",
            "Coarse-channel phase only",
            "Post-channel delay only",
            "PN spectral phase only",
            "Combined example",
            "Combined with PN phase",
            "Custom sliders",
        ],
        value="Combined example",
        label="Choose an effect",
    )
    input_slider = mo.ui.slider(-1000, 1000, step=10, value=180,
                                label="Input-path delay (ns)", show_value=True)
    coarse_slider = mo.ui.slider(-1000, 1000, step=10, value=250,
                                 label="Coarse-channel phase slope (equivalent ns)",
                                 show_value=True)
    post_slider = mo.ui.slider(-1000, 1000, step=10, value=90,
                               label="Post-channel delay (ns)", show_value=True)
    pn_slider = mo.ui.slider(0.0, 2.0, step=0.1, value=0.0,
                             label="Intrinsic PN spectral phase ripple (rad)",
                             show_value=True)
    wrap_checkbox = mo.ui.checkbox(value=False, label="Wrap displayed phase to ±π")
    mo.vstack([
        scenario,
        mo.md("The sliders take effect when **Custom sliders** is selected."),
        mo.hstack([input_slider, coarse_slider], justify="start"),
        mo.hstack([post_slider, pn_slider], justify="start"),
        wrap_checkbox,
    ])
    return scenario, input_slider, coarse_slider, post_slider, pn_slider, wrap_checkbox


@app.cell
def _(coarse_slider, input_slider, pn_slider, post_slider, scenario):
    _presets = {
        "Pure input delay": (350.0, 0.0, 0.0, 0.0),
        "Coarse-channel phase only": (0.0, 350.0, 0.0, 0.0),
        "Post-channel delay only": (0.0, 0.0, 350.0, 0.0),
        "PN spectral phase only": (0.0, 0.0, 0.0, 1.2),
        "Combined example": (180.0, 250.0, 90.0, 0.0),
        "Combined with PN phase": (180.0, 250.0, 90.0, 0.9),
    }
    settings = (_presets[scenario.value]
                if scenario.value in _presets else
                (float(input_slider.value), float(coarse_slider.value),
                 float(post_slider.value), float(pn_slider.value)))
    return (settings,)


@app.cell
def _(COARSE_HZ, FINE_HZ, fitted_slopes, np, phase_terms, pn_tooth_phase, settings):
    channels = np.arange(400, 408)
    fine_offsets = (np.arange(3, 30) - 16) * FINE_HZ
    sky_terms = phase_terms(channels, fine_offsets, *settings[:3])
    sky_centres = phase_terms(channels, np.array([0.0]), *settings[:3])["total"][:, 0]
    pn_phase, pn_intrinsic = pn_tooth_phase(channels, *settings)
    slopes = fitted_slopes(channels, fine_offsets, *settings)
    fine_frequency_mhz = (channels[:, None] * COARSE_HZ + fine_offsets[None, :]) / 1e6
    return (
        channels,
        fine_frequency_mhz,
        fine_offsets,
        pn_intrinsic,
        pn_phase,
        sky_centres,
        sky_terms,
        slopes,
    )


@app.cell(hide_code=True)
def _(mo, settings, slopes):
    _input, _coarse, _post, _pn = settings
    mo.md(f"""
    ## What this setting predicts

    | Phase source | Set value | Effect on centres | Effect within channel |
    |:--|--:|:--|:--|
    | Delay before channelizer | {_input:+.0f} ns | yes | yes |
    | Coarse-channel phase term | {_coarse:+.0f} ns equivalent | yes | no |
    | Delay after channelizer | {_post:+.0f} ns | no | yes |
    | PN comb's own spectral phase | {_pn:.1f} rad | PN teeth only | PN teeth only |

    Ignoring phase-wrap aliases, the sky slopes should be **{_input + _coarse:+.0f} ns
    between centres** and **{_input + _post:+.0f} ns within a channel**.
    The measured simulation gives **{slopes['sky_across_ns']:+.1f} ns** and
    **{slopes['sky_within_ns']:+.1f} ns**. A single delay gives equal numbers;
    unequal numbers can arise from a coarse phase convention, a shift after
    channelization, or both.
    """)
    return


@app.cell
def _(mo):
    selected_channel = mo.ui.slider(400, 407, step=1, value=404,
                                    label="Inspect coarse channel", show_value=True)
    selected_raw_bin = mo.ui.slider(3, 29, step=1, value=22,
                                    label="Inspect retained raw fine bin", show_value=True)
    selected_tooth = mo.ui.slider(0, 3, step=1, value=1,
                                  label="Inspect PN tooth", show_value=True)
    mo.hstack([selected_channel, selected_raw_bin, selected_tooth],
              justify="start")
    return selected_channel, selected_raw_bin, selected_tooth


@app.cell(hide_code=True)
def _(
    COARSE_HZ,
    FINE_HZ,
    PN_OFFSETS_HZ,
    mo,
    np,
    pn_intrinsic,
    pn_phase,
    selected_channel,
    selected_raw_bin,
    selected_tooth,
    sky_terms,
):
    _c = int(selected_channel.value)
    _b = int(selected_raw_bin.value)
    _j = int(selected_tooth.value)
    _ci = _c - 400
    _bi = _b - 3
    _offset_hz = (_b - 16) * FINE_HZ
    _f_c_mhz = _c * COARSE_HZ / 1e6
    _input_rad = sky_terms["input"][_ci, _bi]
    _coarse_rad = sky_terms["coarse"][_ci, _bi]
    _post_rad = sky_terms["post"][_ci, _bi]
    _step2_rad = _input_rad + _coarse_rad
    _sky_rad = _step2_rad + _post_rad
    _wrapped_rad = np.angle(np.exp(1j * _sky_rad))
    _pn_extra_rad = pn_intrinsic[_ci, _j]
    _pn_total_rad = pn_phase[_ci, _j]

    mo.vstack([
        mo.md(r"""
        ## Formula at each step

        Define coarse channel \(c\), retained raw fine bin \(b\), and PN tooth
        \(j\in\{0,1,2,3\}\). Here \(c_0=400\),
        \(\Delta F=781{,}250\,\mathrm{Hz}\),
        \(f_c=c\Delta F\), and
        \(\nu_b=(b-16)\Delta F/27\) is the frequency *inside* the coarse
        channel. All delays in the formulas are converted from ns to seconds.

        **1. Delay before channelization** uses absolute frequency relative
        to the reference centre:
        \[
        \phi_1(c,b)=2\pi\big[(c-c_0)\Delta F+\nu_b\big]\tau_{\rm input}.
        \]

        **2. Coarse-channel phase term** adds a phase at each channel centre:
        \[
        \phi_2(c,b)=\phi_1(c,b)
          +2\pi(c-c_0)\Delta F\tau_{\rm coarse}.
        \]
        The parameter \(\tau_{\rm coarse}\) is an *equivalent slope*, not a
        claim that a second physical delay exists.

        **3. Delay of the channelized baseband stream** adds a slope only in
        frequency relative to its own centre:
        \[
        \phi_3(c,b)=\phi_2(c,b)+2\pi\nu_b\tau_{\rm post}.
        \]
        Thus, before aliasing, a sky signal has apparent slopes
        \(\tau_{\rm input}+\tau_{\rm coarse}\) between centres and
        \(\tau_{\rm input}+\tau_{\rm post}\) within a channel.

        **4. Optional PN generator phase** affects the simulated injected
        comb only. Its four tooth offsets are
        \(\nu_j=-329{,}589.84375+j\times195{,}312.5\,\mathrm{Hz}\):
        \[
        \phi_{\rm PN}(c,j)=\phi_3(c,\nu_j)
          +A\sin\!\left[\frac{2\pi}{9}\big(4(c-c_0)+j\big)+0.4\right].
        \]
        The nine-tooth sinusoid is a teaching example, not a measured model
        of the ADC-box PN generator.
        """),
        mo.md(f"""
        **Numbers at your selected point:** channel **{_c}** has centre
        **{_f_c_mhz:.6f} MHz**; raw fine bin **{_b}** is
        **{_offset_hz / 1e3:+.3f} kHz** from that centre.

        | Step | Phase added (rad) | Running phase (rad) |
        |:--|--:|--:|
        | 1. Input delay | {_input_rad:+.4f} | {_input_rad:+.4f} |
        | 2. Coarse term | {_coarse_rad:+.4f} | {_step2_rad:+.4f} |
        | 3. Post-channel delay | {_post_rad:+.4f} | {_sky_rad:+.4f} |

        The final sky phase wraps to **{_wrapped_rad:+.4f} rad** modulo
        \(2\pi\). At PN tooth **{_j}** ({PN_OFFSETS_HZ[_j] / 1e3:+.3f} kHz),
        the generator contributes **{_pn_extra_rad:+.4f} rad**, giving a PN
        phase of **{_pn_total_rad:+.4f} rad** in this model.
        """),
        mo.md(r"""
        For two receivers \(a,b\), the phase-only visibility correction is
        \[
          V_{ab}^{\rm corrected}=V_{ab}^{\rm observed}
          \exp\!\left[-i\big(\phi_{3,a}-\phi_{3,b}\big)\right].
        \]
        The PN generator term is a possible *measurement contaminant*; it is
        not automatically included in a sky correction.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(FINE_HZ, mo, np, selected_raw_bin, settings):
    _post_ns = settings[2]
    _selected_offset_hz = (int(selected_raw_bin.value) - 16) * FINE_HZ
    _phase_at_selected = 2 * np.pi * _selected_offset_hz * _post_ns * 1e-9
    _phase_at_minus_200 = 2 * np.pi * (-200_000.0) * _post_ns * 1e-9
    _phase_at_plus_200 = 2 * np.pi * 200_000.0 * _post_ns * 1e-9
    _one_output_sample_ns = 1080.0
    _one_sample_phase = 2 * np.pi * _selected_offset_hz * _one_output_sample_ns * 1e-9
    mo.vstack([
        mo.md(r"""
        ## Where does a *post-channelization* phase slope come from?

        Imagine one RF tone at \(f=f_c+\nu\). A simplified coarse channelizer
        shifts \(f_c\) down to zero, leaving a complex baseband tone:
        \[
          x(t)=e^{i2\pi(f_c+\nu)t}
          \quad\longrightarrow\quad
          b_c(t)=x(t)e^{-i2\pi f_ct}=e^{i2\pi\nu t}.
        \]

        Now compare this stream with another FPGA's channelized stream that is
        delayed by \(\Delta t\). At a common comparison time, its baseband
        tone is
        \[
          b_c(t-\Delta t)
          =e^{i2\pi\nu(t-\Delta t)}
          =b_c(t)e^{-i2\pi\nu\Delta t}.
        \]
        The **relative phase is \(-2\pi\nu\Delta t\)**. At the channel centre
        \(\nu=0\), it is zero. Below the centre it has one sign; above the
        centre it has the other. Plotting phase against fine frequency
        therefore gives a straight slope. The notebook uses
        \(+2\pi\nu\tau_{\rm post}\), so its signed parameter is
        \(\tau_{\rm post}=-\Delta t\) in this example.

        By comparison, delaying the *RF signal before channelization* gives
        \(e^{-i2\pi(f_c+\nu)\Delta t}\): a phase at \(f_c\) **and** a slope
        with \(\nu\). This is why the two locations can leave different
        patterns, even though both involve a time offset. A phase caused by
        the PFB's start/derotation convention can add another channel-centre
        term.
        """),
        mo.md(f"""
        **With the current post-channel setting of {_post_ns:+.0f} ns:**

        | Fine offset from centre | Model phase contribution |
        |:--|--:|
        | −200 kHz | {_phase_at_minus_200:+.3f} rad |
        | 0 kHz | 0 rad |
        | +200 kHz | {_phase_at_plus_200:+.3f} rad |
        | Selected bin: {_selected_offset_hz / 1e3:+.3f} kHz | {_phase_at_selected:+.3f} rad |

        One channelized output sample is **1.08 µs** in this system. A residual
        offset of exactly one such sample would give
        **{_one_sample_phase:+.3f} rad** at your selected bin. The large FPGA
        packet offsets are already aligned separately; this control represents
        any *remaining relative offset* in the streams being compared.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(COARSE_HZ, mo, settings, slopes):
    _centre = slopes["sky_across_ns"]
    _inside = slopes["sky_within_ns"]
    _inferred_post = _inside - _centre
    _input, _coarse, _post, _pn = settings
    _alias_ns = 1e9 / COARSE_HZ
    mo.vstack([
        mo.md(r"""
        ## Can the two slopes be split into pre- and post-channel delays?

        **Yes, if those are the only two phase effects.** In that restricted
        model, a delay before channelization changes both frequency axes;
        a delay applied to an already channelized stream changes only its
        fine-frequency axis. Thus
        \[
        S_{\rm centres}=\tau_{\rm pre},\qquad
        S_{\rm inside}=\tau_{\rm pre}+\tau_{\rm post},\qquad
        \boxed{\tau_{\rm post}=S_{\rm inside}-S_{\rm centres}}.
        \]

        The subtraction also works as an **effective correction** even when
        the physical cause is unknown: store the measured phase at each coarse
        centre, then add the measured within-channel slope at fine offset
        \(\nu\), namely
        \(\phi(c,\nu)=\phi_{\rm centre}(c)+2\pi\nu S_{\rm inside}\).
        There is no need to label which circuit supplied each part before
        correcting the sampled frequencies.

        The caveat is *physical attribution*. If the coarse channelizer or
        PN source also adds a centre-to-centre phase slope, then
        \[
        S_{\rm centres}=\tau_{\rm pre}+\tau_{\rm coarse},\qquad
        S_{\rm inside}=\tau_{\rm pre}+\tau_{\rm post},
        \]
        so the inferred ``post delay'' is actually
        \(\tau_{\rm post}-\tau_{\rm coarse}\). It still parameterizes the
        aggregate phase correctly; it no longer tells us where a physical
        delay occurred.
        """),
        mo.md(f"""
        **Current scenario:** fitting the sky phase gives
        \(S_{{centres}}={_centre:+.1f}\) ns and
        \(S_{{inside}}={_inside:+.1f}\) ns. The two-delay assumption would
        infer **pre = {_centre:+.1f} ns** and
        **post = {_inferred_post:+.1f} ns**. The controls actually set input
        delay **{_input:+.1f} ns**, coarse-only term **{_coarse:+.1f} ns**,
        and post-channel delay **{_post:+.1f} ns**. Try *Pure input delay*
        and then *Combined example* to see when the inferred labels are valid.

        The centre slope is ambiguous by **{_alias_ns:.0f} ns** because the
        centres are spaced by 781.25 kHz. An added integer multiple of that
        alias to the inferred pre-delay can be offset by the opposite change
        in the inferred post-delay. Directly measured per-channel phases avoid
        having to choose that unwrap for the aggregate correction.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    COARSE_HZ,
    PN_OFFSETS_HZ,
    channels,
    fine_frequency_mhz,
    fine_offsets,
    mo,
    np,
    pn_phase,
    settings,
    sky_centres,
    sky_terms,
    slopes,
    plt,
    wrap_checkbox,
):
    _fig, _axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    _colours = plt.cm.tab10(np.arange(len(channels)))

    def _display_phase(_phase):
        return np.angle(np.exp(1j * _phase)) if wrap_checkbox.value else _phase

    _pure_input = sky_terms["input"]
    _axes[0].plot(fine_frequency_mhz.ravel(),
                  _display_phase(_pure_input).ravel(),
                  color="0.65", lw=1.4, label="input delay alone")
    for _i, _ch in enumerate(channels):
        _axes[0].plot(fine_frequency_mhz[_i],
                      _display_phase(sky_terms["total"][_i]),
                      ".-", ms=2.5, lw=1.2, color=_colours[_i],
                      label="sky fine bins" if _i == 0 else None)
        _axes[0].scatter(_ch * COARSE_HZ / 1e6,
                         _display_phase(sky_centres[_i]),
                         marker="D", s=24, color="black", zorder=4,
                         label="coarse centres" if _i == 0 else None)
        _axes[0].scatter((_ch * COARSE_HZ + PN_OFFSETS_HZ) / 1e6,
                         _display_phase(pn_phase[_i]),
                         marker="x", s=31, color=_colours[_i], zorder=5,
                         label="PN teeth" if _i == 0 else None)
    _axes[0].set(xlabel="Absolute RF frequency (MHz)",
                 ylabel="Relative phase (rad)",
                 title="Broadband phase: each colour is one coarse channel")
    _axes[0].grid(alpha=.25)
    _axes[0].legend(ncol=4, fontsize=9)

    for _i, _ch in enumerate(channels):
        _sky_local = sky_terms["total"][_i] - sky_centres[_i]
        _pn_local = pn_phase[_i] - np.mean(pn_phase[_i])
        _axes[1].plot(fine_offsets / 1e3, _display_phase(_sky_local),
                      ".-", color=_colours[_i], lw=1.1, ms=2.5,
                      label="sky" if _i == 0 else None)
        _axes[1].scatter(PN_OFFSETS_HZ / 1e3, _display_phase(_pn_local),
                         marker="x", s=25, color=_colours[_i],
                         label="PN teeth" if _i == 0 else None)
    _axes[1].set(xlabel="Offset from each coarse centre (kHz)",
                 ylabel="Phase after removing centre (rad)",
                 title="Zoom into every coarse channel; curves overlap when local slopes agree")
    _axes[1].grid(alpha=.25)
    _axes[1].legend(fontsize=9)

    _names = ["sky: centres", "sky: inside", "PN: centres", "PN: inside"]
    _values = [slopes[_key] for _key in
               ("sky_across_ns", "sky_within_ns", "pn_across_ns", "pn_within_ns")]
    _bars = _axes[2].bar(_names, _values,
                         color=["#355f9b", "#46a2c7", "#a45281", "#db8c9f"])
    _axes[2].axhline(settings[0], color="0.3", ls="--", lw=1.2,
                     label="actual input-path delay")
    _axes[2].set(ylabel="Apparent delay from phase slope (ns)",
                 title="Fit a line to wrapped phase, then convert its slope to delay")
    _axes[2].grid(axis="y", alpha=.25)
    _axes[2].legend(fontsize=9)
    for _bar, _value in zip(_bars, _values):
        _axes[2].text(_bar.get_x() + _bar.get_width() / 2,
                      _value + (15 if _value >= 0 else -15),
                      f"{_value:+.0f}", ha="center",
                      va="bottom" if _value >= 0 else "top", fontsize=9)
    mo.md("**Reading the plots:** A continuous input delay tilts all the sky data. "
          "A coarse-only term moves each coloured segment's centre. A post-channel "
          "delay tilts each segment around its own centre. Coloured crosses are PN "
          "teeth; with PN spectral phase enabled they can depart from the sky curve.")
    _fig
    return


@app.cell(hide_code=True)
def _(COARSE_HZ, PN_OFFSETS_HZ, mo, settings, slopes):
    _coarse_alias_ns = 1e9 / COARSE_HZ
    _tooth_alias_ns = 1e9 / (PN_OFFSETS_HZ[1] - PN_OFFSETS_HZ[0])
    _pn_diff = slopes["pn_across_ns"] - slopes["pn_within_ns"]
    mo.md(f"""
    ## Why the two fitted delays can disagree

    1. **Pure input delay:** select it above. The centre-to-centre and
       within-channel fits agree. This is the signature of a simple path delay.
    2. **Coarse phase term:** it changes phase at the centres without adding a
       slope to the baseband within each channel. The Lambda FPGA has an
       oversampled PFB with a derotation/FFT rotation buffer. A differing
       timing origin between boards *could* create this kind of relative term;
       this notebook does not assert that it does.
    3. **Post-channel delay:** it adds a phase slope in baseband fine frequency
       but leaves the centre of each baseband channel alone.
    4. **PN source phase:** choose the PN-only scenario. The toy nine-tooth
       sinusoid is an intentionally arbitrary phase of the generator's Fourier
       coefficients, not a model of the real ADC-box PN. It alters the PN fits
       while the sky fits remain unchanged. In this setting the two PN slopes
       differ by **{_pn_diff:+.1f} ns**.

    **Phase wrapping matters.** Adjacent coarse centres are {COARSE_HZ / 1e3:.2f}
    kHz apart, so their slope is ambiguous by **{_coarse_alias_ns:.0f} ns**.
    The simulated PN teeth are {(PN_OFFSETS_HZ[1] - PN_OFFSETS_HZ[0]) / 1e3:.2f}
    kHz apart, giving a **{_tooth_alias_ns:.0f} ns** alias. Switch on wrapped
    phase above and increase a delay beyond half an alias interval to see the
    fit jump. A jump in a fitted delay need not mean the hardware delay jumped.

    **How to distinguish causes on the instrument:** repeat the phase
    measurement with PN toggled or restarted; inject a common stepped tone
    across a coarse boundary; compare same-board and cross-board baselines;
    and test whether the correction predicts held-out sky visibilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Relation to the October PN measurements

    The real PN comb has a selected family of teeth spaced 195.3125 kHz, four
    per 781.25 kHz coarse-channel step. Its fitted coarse-centre and local
    slopes differ by hundreds of nanoseconds on some boards. That establishes
    a repeatable *phase pattern*, not two identified hardware delays. The

    | FPGA | Apparent slope across coarse centres | Apparent local PN slope |
    |:--|--:|--:|
    | 0 | −25 ns | −56 ns |
    | 1 | +343 ns | −3 ns |
    | 2 | +40 ns | +284 ns |
    | 3 | +629 ns | +397 ns |

    The centre slopes are from the first PN run and the local slopes from a
    longer independent run. Coarse-centre slopes also have a 1280 ns alias.
    See `analysis/results/pn_adc_20261003/FULL_REPORT.md` for repeatability
    and quality limits. This notebook isolates possible mechanisms so their
    signatures are easier to recognize. For the general DSP concepts, see
    [channelizer phase conventions](https://www.mathworks.com/help/dsphdl/ref/dsphdl.channelizer-system-object.html)
    and [group delay as phase slope](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html).
    """)
    return


if __name__ == "__main__":
    app.run()
