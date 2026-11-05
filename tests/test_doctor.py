from trainkit.doctor.probe import GPU, SystemInfo, parse_cuda_driver_version, parse_nvidia_query
from trainkit.doctor.report import analyze, nvidia_driver_blocked, recommended_profile, torch_backend_for, verdicts
from trainkit.recipes import all_recipes


def info(gpus=(), ram=16.0, disk=100.0):
    return SystemInfo(os="Windows", os_version="10", arch="AMD64", cpu="x", cpu_count=8, ram_gb=ram, disk_free_gb=disk,
                      user_home="C:\\Users\\a", trainkit_home="C:\\Users\\a\\.trainkit", python="3.12", uv="0.9",
                      gpus=list(gpus))


def nv(name="NVIDIA GeForce RTX 3060", vram=12.0, driver="560.94", cc="8.6"):
    return GPU("nvidia", name, vram, driver, "12.6", cc)


def test_parse_nvidia_query():
    gpus = parse_nvidia_query("NVIDIA GeForce RTX 4070, 12282, 566.36, 8.9\n")
    assert gpus[0].name == "NVIDIA GeForce RTX 4070" and gpus[0].vram_gb == 12.0
    assert gpus[0].driver == "566.36" and gpus[0].compute_cap == "8.9"
    old = parse_nvidia_query("GeForce GTX 1060 6GB, 6144, 472.12\n")
    assert old[0].compute_cap == "" and old[0].vram_gb == 6.0


def test_parse_cuda_version():
    assert parse_cuda_driver_version("| NVIDIA-SMI 566.36  Driver Version: 566.36  CUDA Version: 12.7 |") == "12.7"
    assert parse_cuda_driver_version("") == ""


def test_backend_selection():
    assert torch_backend_for(info([nv()])) == "auto"
    assert torch_backend_for(info([nv("GeForce GTX 1080", 8, "580.10", "6.1")])) == "cu126"  # CUDA 13 dropped Pascal
    assert torch_backend_for(info([nv("Quadro K2000", 2, "470.00", "3.0")])) == "cpu"
    assert torch_backend_for(info([])) == "auto"


def test_blackwell_needs_new_driver():
    rtx50 = info([nv("NVIDIA GeForce RTX 5070", 12, "566.00", "12.0")])
    assert nvidia_driver_blocked(rtx50)
    assert recommended_profile(rtx50) == "preview"
    assert not nvidia_driver_blocked(info([nv("NVIDIA GeForce RTX 5070", 12, "576.02", "12.0")]))


def test_verdicts_by_tier():
    cpu = {v.recipe_id: v for v in verdicts(info([]), all_recipes())}
    assert cpu["yolo-detect"].train == "no" and cpu["yolo-detect"].preview == "ok"
    small = {v.recipe_id: v for v in verdicts(info([nv(vram=4.0)]), all_recipes())}
    assert small["yolo-detect"].train == "ok"
    assert small["rtdetr"].train == "warn"
    mac = SystemInfo(**{**info().__dict__, "os": "macOS", "gpus": [GPU("apple", "Apple M2", 16)]})
    assert {v.recipe_id: v for v in verdicts(mac, all_recipes())}["yolo-detect"].train == "ok"


def test_analyze_reports_old_driver_and_disk():
    f = analyze(info([nv(driver="440.10")], disk=3))
    levels = [x.level for x in f]
    assert "error" in levels
    assert any("disk" in x.title.lower() or "GB disk" in x.title for x in f)
