import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("app.services.ffmpeg")

# Common static/embedded Windows binary locations
KNOWN_BIN_DIRS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "Lib" / "site-packages" / "static_ffmpeg" / "bin" / "win32",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python314" / "Lib" / "site-packages" / "static_ffmpeg" / "bin" / "win32",
    Path("D:/gay/VEO_3_create-16-10/creatve_viodeo"),
    Path("C:/ffmpeg/bin"),
    Path("C:/Program Files/ffmpeg/bin"),
]


def _find_binary(bin_name: str, env_var: str) -> Optional[str]:
    """Locate binary from env/settings, PATH, static_ffmpeg, or known paths."""
    # 1. Direct env variable or setting
    custom = os.getenv(env_var, "").strip().strip("'\"")
    if not custom:
        try:
            from dotenv import load_dotenv
            from app.database import BASE_DIR
            load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
            custom = os.getenv(env_var, "").strip().strip("'\"")
        except Exception:
            pass
    if custom and Path(custom).exists():
        return str(Path(custom).resolve())

    # 2. System PATH
    found = shutil.which(bin_name)
    if found:
        return str(Path(found).resolve())

    # 3. static_ffmpeg dynamically if installed
    try:
        import static_ffmpeg
        # Try to resolve bin folder from package
        pkg_dir = Path(static_ffmpeg.__file__).resolve().parent / "bin" / "win32"
        candidate = pkg_dir / f"{bin_name}.exe"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    # 4. Known directories
    exe_name = f"{bin_name}.exe" if os.name == "nt" else bin_name
    for d in KNOWN_BIN_DIRS:
        cand = d / exe_name
        if cand.exists():
            return str(cand.resolve())

    # 5. imageio_ffmpeg fallback (ffmpeg only)
    if bin_name == "ffmpeg":
        try:
            import imageio_ffmpeg
            cand = imageio_ffmpeg.get_ffmpeg_exe()
            if cand and Path(cand).exists():
                return str(Path(cand).resolve())
        except Exception:
            pass

    return None


def get_ffmpeg_path() -> Optional[str]:
    """Retrieve absolute path to FFmpeg executable."""
    return _find_binary("ffmpeg", "FFMPEG_PATH")


def get_ffprobe_path() -> Optional[str]:
    """Retrieve absolute path to ffprobe executable."""
    return _find_binary("ffprobe", "FFPROBE_PATH")


def check_ffmpeg_available() -> Dict[str, Any]:
    """Check availability and version of FFmpeg and ffprobe."""
    ffmpeg_exe = get_ffmpeg_path()
    ffprobe_exe = get_ffprobe_path()

    ffmpeg_ver = ""
    ffprobe_ver = ""
    ffmpeg_ok = False
    ffprobe_ok = False

    if ffmpeg_exe:
        try:
            res = subprocess.run([ffmpeg_exe, "-version"], capture_output=True, text=True, timeout=10.0)
            if res.returncode == 0:
                ffmpeg_ok = True
                ffmpeg_ver = res.stdout.splitlines()[0] if res.stdout else "unknown"
        except Exception as e:
            logger.warning(f"Error running ffmpeg -version: {e}")

    if ffprobe_exe:
        try:
            res = subprocess.run([ffprobe_exe, "-version"], capture_output=True, text=True, timeout=10.0)
            if res.returncode == 0:
                ffprobe_ok = True
                ffprobe_ver = res.stdout.splitlines()[0] if res.stdout else "unknown"
        except Exception as e:
            logger.warning(f"Error running ffprobe -version: {e}")

    return {
        "ffmpeg_available": ffmpeg_ok,
        "ffmpeg_path": ffmpeg_exe,
        "ffmpeg_version": ffmpeg_ver,
        "ffprobe_available": ffprobe_ok,
        "ffprobe_path": ffprobe_exe,
        "ffprobe_version": ffprobe_ver,
        "ready": ffmpeg_ok and ffprobe_ok
    }


def run_ffmpeg(args: List[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    """
    Execute FFmpeg with argument array.
    Safe execution:
    - Never uses shell=True
    - Validates binary exists
    - Captures stdout/stderr safely
    """
    exe = get_ffmpeg_path()
    if not exe:
        raise RuntimeError("FFmpeg executable not found. Please install FFmpeg or set FFMPEG_PATH in Settings.")

    cmd = [exe] + args
    logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")

    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
    except subprocess.TimeoutExpired as te:
        raise RuntimeError(f"FFmpeg operation timed out after {timeout} seconds: {te}")
    except Exception as e:
        raise RuntimeError(f"Failed to execute FFmpeg: {e}")


def run_ffprobe(args: List[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Execute ffprobe with argument array safely."""
    exe = get_ffprobe_path()
    if not exe:
        raise RuntimeError("ffprobe executable not found. Please install ffprobe or set FFPROBE_PATH in Settings.")

    cmd = [exe] + args
    logger.debug(f"Running ffprobe command: {' '.join(cmd)}")

    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
    except subprocess.TimeoutExpired as te:
        raise RuntimeError(f"ffprobe operation timed out after {timeout} seconds: {te}")
    except Exception as e:
        raise RuntimeError(f"Failed to execute ffprobe: {e}")
