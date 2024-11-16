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
    life_pieces=0xF5868,
    bombs=0xF5870,
    bomb_pieces=0xF5874,
    bonus_count=0xF5894,
    power=0xF5858,
    piv=0xF584C,
    graze=0xF5840,
)


# get a handle to the game window
_game_windows = gw.getWindowsWithTitle(_GAME_TITLE)

if len(_game_windows) == 1:
    logger.info(f"Found game window: {_game_windows[0]}")
else:
    logger.error(f"Cannot find the window with title: {_GAME_TITLE}")
    exit(1)

_game_window = _game_windows[0]

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


def _suspend_game_process():
    ctypes.windll.kernel32.DebugActiveProcess(_game_pid)


def _resume_game_process():
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


def _init():
    """
    Enter the "practice start" phase from the title screen
    """
    _get_focus()

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
    _press_and_release("z")
    time.sleep(1)
    _press_and_release("z")
    time.sleep(2.5)

    _suspend_game_process()


def read_game_status_int(key: str):
    offset = _OFFSETS[key]
    return _read_game_memory(offset, 4)


if __name__ == "__main__":
    try:
        _init()
        while True:
            status = {}
            for v in _OFFSETS:
                v_offset = _OFFSETS[v]
                data = _read_game_memory(v_offset, 4)
                if data is not None:
                    status[v] = int.from_bytes(data, byteorder="little")
            logger.info(status)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Quitting...")
    finally:
        _resume_game_process()
        win32api.CloseHandle(_process_handle)
