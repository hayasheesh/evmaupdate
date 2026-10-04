from tools.Utils import dispatch_background_color


def test_dispatch_background_uses_green_for_charge_and_red_for_discharge():
    color, alpha = dispatch_background_color(1.0, intensity=1.0)
    assert color == "lightgreen"
    assert alpha > 0.0
    color, alpha = dispatch_background_color(-1.0, intensity=1.0)
    assert color == "lightcoral"
    assert alpha > 0.0
    color, alpha = dispatch_background_color(0.0)
    assert color is None
    assert alpha == 0.0


def test_dispatch_background_alpha_scales_with_intensity_not_sign():
    _, alpha_min = dispatch_background_color(5.0, intensity=0.0)
    _, alpha_mid = dispatch_background_color(5.0, intensity=0.5)
    _, alpha_max = dispatch_background_color(5.0, intensity=1.0)
    assert alpha_min < alpha_mid < alpha_max
    # Sign only picks the color; a small discharge is not darker than a
    # large discharge just because it is negative.
    _, alpha_small_discharge = dispatch_background_color(-1.0, intensity=0.1)
    _, alpha_big_discharge = dispatch_background_color(-100.0, intensity=0.9)
    assert alpha_small_discharge < alpha_big_discharge


def test_dispatch_background_intensity_is_clamped():
    _, alpha_over = dispatch_background_color(5.0, intensity=5.0)
    _, alpha_at_max = dispatch_background_color(5.0, intensity=1.0)
    assert alpha_over == alpha_at_max
