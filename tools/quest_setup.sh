#!/usr/bin/env bash
# Prepare a connected Quest for unattended runs: it behaves as if worn,
# stays awake on USB power, and (optionally) listens for adb over Wi-Fi.
# Horizon OS will not start an OpenXR activity while the proximity sensor
# reports the headset off, so without this every run needs a person wearing
# it. Everything here resets on reboot; `--restore` undoes it sooner.
#
# Usage: tools/quest_setup.sh [--wifi] [--restore]
# Set ANDROID_SERIAL when more than one device (or the same headset over USB
# and Wi-Fi) is attached.
set -euo pipefail
export MSYS_NO_PATHCONV=1 # Git Bash on Windows would rewrite /sdcard paths

wifi=0
restore=0
for arg in "$@"; do
    case "$arg" in
        --wifi) wifi=1 ;;
        --restore) restore=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

adb get-state >/dev/null
if [ "$restore" = 1 ]; then
    adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable >/dev/null
    adb shell svc power stayon false
    echo "proximity sensor and sleep restored"
    exit 0
fi

echo "$(adb shell getprop ro.product.model) | Android $(adb shell getprop ro.build.version.release)" \
    "| Horizon OS $(adb shell getprop ro.vros.build.version) | $(adb shell getprop ro.soc.model)"

# Pretend the headset is on a head until automation_disable or reboot.
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close >/dev/null
adb shell svc power stayon usb
echo "proximity: simulated worn; stays awake on USB"

# The default 256 KiB log ring loses the app's startup lines to Blade's
# per-pass descriptor-pool messages within seconds.
adb logcat -G 16M

if adb shell pm list packages | grep -q 'rust.red_eye_xr'; then
    adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA
    adb shell pm grant rust.red_eye_xr com.oculus.permission.USE_SCENE
    echo "granted HEADSET_CAMERA and USE_SCENE (the headset's depth) to rust.red_eye_xr"
fi

if [ "$wifi" = 1 ]; then
    ip=$(adb shell ip -f inet addr show wlan0 | sed -n 's/.*inet \([0-9.]*\)\/.*/\1/p')
    adb tcpip 5555
    sleep 3
    adb connect "$ip:5555"
    echo "wireless adb at $ip:5555 (until reboot)"
fi

adb shell dumpsys battery | sed -n 's/^ *\(level\|temperature\): /\1 /p' | tr '\n' ' '
echo
