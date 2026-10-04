"""Keep a long run off the slow Windows scheduling path.

An episode is hundreds of short CPU-to-GPU waits, so this workload is decided by
scheduling latency rather than by arithmetic, and Windows has three levers that
all move against a process it reads as background. Every call here is
per-process and is released when the process exits; no system setting is
touched.

This lived in pre_train.py and so applied to pretraining only. A fine-tune is
the same workload and ran at normal priority.
"""

from __future__ import annotations

import os

HIGH_PRIORITY_CLASS = 0x00000080
PROCESS_POWER_THROTTLING = 4
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4


def hold_scheduler_priority(label: str = "run") -> None:
    """Ask for high priority, a 1 ms timer, and no power throttling.

    High rather than above-normal: this box runs the trainer for days with
    nothing else on it, and above-normal still sits below a foreground
    application's boosted threads. Real time is the class above this one and is
    not wanted -- it outranks kernel threads and can starve the machine.
    """

    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
        kernel32 = ctypes.windll.kernel32
        # GetCurrentProcess returns the pseudo-handle -1. Without these, ctypes
        # hands it back as a 32-bit int and passes it to SetPriorityClass as
        # one, which arrives as a different 64-bit value: the call failed with
        # ERROR_INVALID_HANDLE on every run for months, and because the return
        # value was never read it failed in silence.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        kernel32.GetPriorityClass.restype = ctypes.c_uint32

        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS):
            raise OSError(f"SetPriorityClass failed with {ctypes.get_last_error()}")
        applied = kernel32.GetPriorityClass(handle)
        if applied != HIGH_PRIORITY_CLASS:
            raise OSError(f"priority class is {applied:#x}, not high")

        # Priority decides which thread gets a core. On a hybrid part it does
        # not decide which kind of core, nor at what frequency, and this box is
        # one: sixteen logical processors at efficiency class 1 and sixteen at
        # class 0. The run is one thread at 100% of one core, so being moved to
        # an efficiency core costs it directly, and "the window is minimised" is
        # one of the signals Windows reads as background. Opting out of power
        # throttling is a separate call.
        #
        # The second bit matters as much: Windows 11 ignores a background
        # process's timeBeginPeriod, and the line above asked for 1 ms because
        # an episode is hundreds of short waits. At the 15.6 ms default every
        # one of them rounds up.
        #
        # Not fatal when it fails. The API is Windows 10 1709 and later, and a
        # machine without it should still keep the priority it just set.
        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_uint32),
                        ("ControlMask", ctypes.c_uint32),
                        ("StateMask", ctypes.c_uint32)]

        kernel32.SetProcessInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
        ]
        kernel32.SetProcessInformation.restype = ctypes.c_int
        state = _PowerThrottlingState(
            Version=1,
            ControlMask=(PROCESS_POWER_THROTTLING_EXECUTION_SPEED
                         | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION),
            # Zero against a set control bit reads as "never throttle me" and
            # "always honour my timer resolution".
            StateMask=0,
        )
        unthrottled = bool(kernel32.SetProcessInformation(
            handle, PROCESS_POWER_THROTTLING, ctypes.byref(state),
            ctypes.sizeof(state),
        ))
        # SetProcessInformation returning true is the whole confirmation
        # available: GetProcessInformation answers 0/0 whether throttling was
        # set on, set off, or never set, so it cannot witness this.
        print(
            f"[{label}] scheduling: high priority, 1 ms timer, "
            f"power throttling opt-out {'applied' if unthrottled else 'REFUSED'}",
            flush=True,
        )
    except Exception as exc:  # a scheduling hint is never worth failing a run for
        print(f"[{label}] scheduler hint unavailable: {exc}", flush=True)
