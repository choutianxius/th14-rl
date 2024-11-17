import sys
import ctypes
import win32api
import win32con
import win32process
import logging
import time
import os
import pygetwindow as gw
import keyboard
import pyautogui
from collections import deque


logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("interface")


_MODULE_NAME = "th14.exe"
_GAME_TITLE = "Double Dealing Character. ver 1.00b"
_FRAME_RATE = 60

_OFFSETS = dict(
    score=0xF5830,
    lives=0xF5864,
    life_fragments=0xF5868,  # 3 life fragments = 1 life
    bombs=0xF5870,
    bomb_fragments=0xF5874,  # 8 bomb fragments = 1 bomb
    bonus_count=0xF5894,
    power=0xF5858,
    piv=0xF584C,
    graze=0xF5840,
    game_state=0xF7AC8,  # 0 = pausing, 1 = end of run, 2 = playing
    in_dialog=0xF7BA8,  # -1 = in dialog, otherwise = not
)

# the Windows borders are included in _WINDOW_WIDTH and _WINDOW_HEIGHT
# tested on Win11 with 2560x1440 screen with 100% scale
# TODO: programmatically get the "inner" window dimensions
_WINDOW_WIDTH = 646
_WINDOW_HEIGHT = 509
FRAME_WIDTH = _WINDOW_WIDTH - 261
FRAME_HEIGHT = _WINDOW_HEIGHT - 60
_FRAME_LEFT = 35
_FRAME_TOP = 42


# get a handle to the game window
_game_windows = gw.getWindowsWithTitle(_GAME_TITLE)

if len(_game_windows) == 1:
    logger.info(f"Found game window: {_game_windows[0]}")
else:
    logger.error(f"Cannot find the window with title: {_GAME_TITLE}")
    exit(1)

_game_window = _game_windows[0]
if _game_window.width != _WINDOW_WIDTH or _game_window.height != _WINDOW_HEIGHT:
    logger.error(
        f"Invalid window resolution: {_game_window.width}x{_game_window.height}"
    )
    logger.info(f"Launch the game with {_WINDOW_WIDTH}x{_WINDOW_HEIGHT} resolution")
    exit(1)

# get the game pid from the window handle
_pid = ctypes.c_ulong()
ctypes.windll.user32.GetWindowThreadProcessId(_game_window._hWnd, ctypes.byref(_pid))
_game_pid = _pid.value


# get the program's base address from the game pid
_base_address = None
_module_handle = win32api.OpenProcess(
    win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, _game_pid
)
_module_list = win32process.EnumProcessModules(_module_handle)

for module in _module_list:
    module_info = win32process.GetModuleFileNameEx(_module_handle, module)
    module_base_name = os.path.basename(module_info)
    if _MODULE_NAME.lower() == module_base_name.lower():
        _base_address = module
        break

if _base_address is not None:
    logger.info(f"Base address of the process main module: {hex(_base_address)}")
    win32api.CloseHandle(_module_handle)
else:
    logger.error("Module base address not found")
    win32api.CloseHandle(_module_handle)
    exit(1)

# create the process handle from the game pid
# the process handle is basically the same as the module handle
# they are created using different APIs for convenience
_PROCESS_VM_READ = 0x0010
_PROCESS_QUERY_INFORMATION = 0x0400
_process_handle = ctypes.windll.kernel32.OpenProcess(
    _PROCESS_VM_READ | _PROCESS_QUERY_INFORMATION, False, _game_pid
)


def suspend_game_process():
    ctypes.windll.kernel32.DebugActiveProcess(_game_pid)


def resume_game_process():
    ctypes.windll.kernel32.DebugActiveProcessStop(_game_pid)


def _read_game_memory(offset, size):
    buffer = ctypes.create_string_buffer(size)
    bytesRead = ctypes.c_int()
    ok = ctypes.windll.kernel32.ReadProcessMemory(
        _process_handle,
        ctypes.c_uint64(_base_address + offset),
        buffer,
        size,
        ctypes.byref(bytesRead),
    )
    if not ok:
        logger.error(
            f"Failed to read memory at offset {hex(offset)}. Process may have exitted."
        )
        return None
    return buffer.raw


def _press_and_release(key):
    keyboard.press(key)
    time.sleep(1 / _FRAME_RATE)
    keyboard.release(key)
    time.sleep(1 / _FRAME_RATE)


def _get_focus():
    _game_window.activate()
    time.sleep(0.2)


def _resume_shooting():
    keyboard.release("z")
    time.sleep(1 / _FRAME_RATE)
    keyboard.press("z")


def init():
    """
    Enter the "practice start" phase from the title screen
    """
    _get_focus()
    release_all_keys()

    # after some time of inactivity, the game will enter a demo play phase
    # and we quit from that
    _press_and_release("down")
    _press_and_release("esc")
    time.sleep(3)

    # now we are on the title screen
    _press_and_release("down")
    _press_and_release("esc")

    # now the cursor is at the last line in the menu
    _press_and_release("down")
    _press_and_release("down")
    _press_and_release("z")
    time.sleep(1)

    # now we are on the game difficulty and character selection screen
    # notice that the character and difficulty are memorized by the game
    # and we assume that the proper selections have been made in previous runs
    _press_and_release("z")
    time.sleep(1)
    _press_and_release("z")
    time.sleep(1)
    _press_and_release("right")
    time.sleep(1)
    _press_and_release("z")
    time.sleep(1)
    _press_and_release("z")
    time.sleep(2.5)

    # always fire
    _resume_shooting()


def read_game_status_int(key: str):
    if key not in _OFFSETS:
        raise ValueError(f"Invalid offset key: {key}")
    offset = _OFFSETS[key]
    return int.from_bytes(_read_game_memory(offset, 4), byteorder="little", signed=True)


def capture_frame():
    """
    Capture and return the current game scene as a Pillow image.
    """
    return pyautogui.screenshot(
        region=(
            _game_window.left + _FRAME_LEFT,
            _game_window.top + _FRAME_TOP,
            FRAME_WIDTH,
            FRAME_HEIGHT,
        )
    )


def skip_dialog():
    """
    Skip the dialog phases
    """
    # a rough estimate of the duration of the dialog to prevent infinite loop
    max_retry = 3 * _FRAME_RATE
    tries = 0
    q = deque(maxlen=3)

    def dialog_end():
        return len(q) == 3 and sum([int(x != -1) for x in q]) == 3

    release_all_keys()
    time.sleep(5 / _FRAME_RATE)
    keyboard.press("ctrl")
    while tries < max_retry:
        t0 = time.time()

        if dialog_end():
            break
        tries += 1
        q.append(read_game_status_int("in_dialog"))
        time.sleep(max(0, 5 / _FRAME_RATE - time.time() + t0))
    else:
        raise Exception("Failed to skip dialog!")
    keyboard.release("ctrl")
    _resume_shooting()


_pressed_keys = {
    "left": False,
    "right": False,
    "up": False,
    "down": False,
    "shift": False,
}


def act(move: int, slow: int, k: int = 1) -> None:
    """
    Perform one action and advance k frames.

    Args
    ----
    move : int
        0 - no op;
        1 - left;
        2 - right;
        3 - up;
        4 - down.
    slow : int
        0 - normal speed;
        1 - slow mode.
    k : int
        Number of frames to advance. The provided action is kept for the k frames.
    """
    if k < 1:
        raise ValueError(f"Invalid k {k}, should be positive")
    t0 = time.time()
    _maintain_keyboard_move(move)
    _maintain_keyboard_slow(slow)
    time.sleep(max(0, k / _FRAME_RATE - time.time() + t0))


def _maintain_keyboard_move(move: int):
    if move == 0:
        for k in ("left", "right", "up", "down"):
            if _pressed_keys[k]:
                keyboard.release(k)
                _pressed_keys[k] = False
    elif 1 <= move <= 4:
        for i, x in enumerate(("left", "right", "up", "down")):
            if i == move - 1:
                if not _pressed_keys[x]:
                    keyboard.press(x)
                    _pressed_keys[x] = True
            else:
                if _pressed_keys[x]:
                    keyboard.release(x)
                    _pressed_keys[x] = False
    else:
        raise ValueError(f"Invalid move flag {move}, should be 0 - 4")


def _maintain_keyboard_slow(slow: int):
    if slow == 0:
        if _pressed_keys["shift"]:
            keyboard.release("shift")
            _pressed_keys["shift"] = False
    elif slow == 1:
        if not _pressed_keys["shift"]:
            keyboard.press("shift")
            _pressed_keys["shift"] = True
    else:
        raise ValueError(f"Invalid slow flag {slow}, should be 0 or 1")


def release_all_keys() -> None:
    for k in _pressed_keys:
        if _pressed_keys[k]:
            keyboard.release(k)
            _pressed_keys[k] = False
    keyboard.release("z")
    keyboard.release("r")
    keyboard.release("esc")
    keyboard.release("ctrl")


def reset_from_end_of_run() -> None:
    """
    Reset when the game is cleared or all lives are lost.
    """
    _press_and_release("up")
    time.sleep(30 / _FRAME_RATE)
    _press_and_release("z")
    time.sleep(30 / _FRAME_RATE)
    count = 0
    max_retry = (3 * _FRAME_RATE) // 5
    while count < max_retry:
        t0 = time.time()
        if read_game_status_int("game_state") == 2:
            break
        count += 1
        time.sleep(max(0, 5 / _FRAME_RATE - time.time() + t0))
    _resume_shooting()


def force_reset() -> None:
    """
    Reset when the game is still running.
    """
    release_all_keys()
    _press_and_release("esc")
    time.sleep(30 / _FRAME_RATE)
    _press_and_release("r")
    time.sleep(30 / _FRAME_RATE)
    count = 0
    max_retry = (3 * _FRAME_RATE) // 5
    while count < max_retry:
        t0 = time.time()
        if read_game_status_int("game_state") == 2:
            break
        count += 1
        time.sleep(max(0, 5 / _FRAME_RATE - time.time() + t0))

    _resume_shooting()


def clean_up():
    resume_game_process()
    release_all_keys()
    win32api.CloseHandle(_process_handle)
    logger.info("Interface successfully exited")


if __name__ == "__main__":
    try:
        init()
        while True:
            t0 = time.time()
            info = {}
            for k in _OFFSETS:
                info[k] = read_game_status_int(k)
            logger.info(info)
            time.sleep(max(0, 30 / _FRAME_RATE - time.time() + t0))
    except KeyboardInterrupt:
        logger.info("Quitting...")
    except Exception as e:
        logger.error("Unexpected error happened")
        logger.error(e)
    finally:
        clean_up()
