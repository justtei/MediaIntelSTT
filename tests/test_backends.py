from backends import _candidates


def test_auto_tries_accelerator_then_cpu():
    assert list(_candidates("auto", cpu_name="CPU", primary_name="GPU")) == ["GPU", "CPU"]
    assert list(_candidates("auto", cpu_name="cpu", primary_name="cuda")) == ["cuda", "cpu"]


def test_explicit_device_falls_back_to_cpu():
    assert list(_candidates("GPU", cpu_name="CPU", primary_name="GPU")) == ["GPU", "CPU"]
    assert list(_candidates("cuda", cpu_name="cpu", primary_name="cuda")) == ["cuda", "cpu"]


def test_explicit_cpu_has_no_duplicate_fallback():
    assert list(_candidates("CPU", cpu_name="CPU", primary_name="GPU")) == ["CPU"]
    assert list(_candidates("cpu", cpu_name="cpu", primary_name="cuda")) == ["cpu"]


def test_none_defaults_to_auto():
    assert list(_candidates(None, cpu_name="CPU", primary_name="GPU")) == ["GPU", "CPU"]
