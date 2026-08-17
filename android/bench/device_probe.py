"""Device state probes over adb (Pixel-class Android).

Thermal vocabulary: Android reports an integer status (0 = THERMAL_STATUS_NONE
… 6 = SHUTDOWN). We map 0 -> "nominal" so the repo's nominal-gate tooling works
unchanged, and record the raw integer alongside — the mapping is disclosed in
methodology/android.md, not silent.
"""
import re
import subprocess

THERMAL_NAMES = {0: "nominal", 1: "light", 2: "moderate", 3: "severe",
                 4: "critical", 5: "emergency", 6: "shutdown"}


def adb(args, serial=None, timeout=30):
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    return subprocess.check_output(cmd, text=True, timeout=timeout)


def getprop(name, serial=None):
    return adb(["shell", "getprop", name], serial).strip()


def device_info(serial=None):
    return {
        "modelIdentifier": getprop("ro.product.model", serial),          # "Pixel 8a"
        "systemName": "Android",
        "systemVersion": getprop("ro.build.version.release", serial),
        "securityPatch": getprop("ro.build.version.security_patch", serial),
        "soc": getprop("ro.soc.model", serial),                          # "Tensor G3"
        "product": getprop("ro.product.device", serial),                 # "akita"
    }


def battery(serial=None):
    out = adb(["shell", "dumpsys", "battery"], serial)
    level = re.search(r"level: (\d+)", out)
    status = re.search(r"status: (\d+)", out)
    plugged = re.search(r"(AC|USB|Wireless) powered: true", out)
    return {
        "batteryLevel": int(level.group(1)) / 100 if level else None,
        # dumpsys status: 2=charging 3=discharging 4=not-charging 5=full
        "batteryState": ("charging" if plugged else "unplugged"),
        "rawStatus": int(status.group(1)) if status else None,
    }


def thermal_status(serial=None):
    out = adb(["shell", "dumpsys", "thermalservice"], serial)
    m = re.search(r"Thermal Status: (\d+)", out)
    raw = int(m.group(1)) if m else None
    return raw, THERMAL_NAMES.get(raw, f"unknown({raw})")
