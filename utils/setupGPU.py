import os
import platform
import subprocess
import shutil

INTEL_GPU = True

if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"
    
def setup_gpu():
    system = platform.system()
    print(f"Detected OS: {system}")

    if system == "Darwin":
        print("macOS detected — using Core OpenGL (CGL).")
        os.environ["MUJOCO_GL"] = "cgl"
        return

    if system == "Linux":
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path:
            try:
                subprocess.run([nvidia_smi_path, "-L"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("NVIDIA GPU detected — using EGL.")
                os.environ["MUJOCO_GL"] = "egl"
                return
            except subprocess.CalledProcessError:
                pass
        print("No NVIDIA GPU — using OSMesa (software).")
        os.environ["MUJOCO_GL"] = "osmesa"
        return

    print("Unknown OS — using OSMesa.")
    os.environ["MUJOCO_GL"] = "osmesa"

def add_ICD_config():
    """
    Attempt to write an NVIDIA ICD config for libglvnd (Linux only).
    This file typically requires root privileges. We try to create it only on Linux
    and catch permission errors — if we can't write it, just warn.
    """
    if platform.system() != "Linux":
        return

    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        print(f"ICD config already exists at {NVIDIA_ICD_CONFIG_PATH}.")
        return

    icd_json = """{
  "file_format_version" : "1.0.0",
  "ICD" : {
    "library_path" : "libEGL_nvidia.so.0"
  }
}
"""
    try:
        # Attempt to create directories if missing
        os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write(icd_json)
        print(f"Wrote NVIDIA ICD config to {NVIDIA_ICD_CONFIG_PATH}.")
    except PermissionError:
        print(f"Permission denied: cannot write {NVIDIA_ICD_CONFIG_PATH}. Run as root to install ICD, or skip this step.")
    except Exception as e:
        print(f"Unexpected error writing ICD config: {e}")


def mujoco_EGL_rendering():
    if platform.system() != "Linux":
        print("EGL rendering not supported on this OS. Skipping.")
        return
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["MUJOCO_GL"] = "egl"
        print("Set MUJOCO_GL='egl' for GPU-backed rendering.")
    except Exception:
        print("NVIDIA GPU not available, skipping EGL rendering.")

def setup_XLA_Triton():
    """
    Optionally set XLA flags to enable Triton GEMM. Harmless if XLA not used,
    but only useful for GPU-backed XLA computations.
    """
    xla_flags = os.environ.get("XLA_FLAGS", "")
    flag_to_add = "--xla_gpu_triton_gemm_any=True"
    if flag_to_add not in xla_flags:
        if xla_flags and not xla_flags.endswith(" "):
            xla_flags += " "
        xla_flags += flag_to_add
        os.environ["XLA_FLAGS"] = xla_flags
        print("Appended XLA flag for Triton GEMM (XLA_FLAGS updated).")
    else:
        print("XLA Triton flag already present.")

def setup_intel_gpu():
    # Set up environment for Intel integrated GPU (e.g., UHD Graphics 620)
    print('Setting up Intel GPU (UHD Graphics 620)...')
    os.environ['MUJOCO_GL'] = 'osmesa'  # Use OSMesa for software rendering
    print('Set MUJOCO_GL=osmesa for Intel GPU.')

def check_gpu_connection():
    """
    Lightweight GPU presence check: return True if NVIDIA tools are available and responding.
    Don't call this on macOS (no nvidia-smi there).
    """
    if platform.system() != "Linux":
        print("Skipping NVIDIA connectivity check (non-Linux system).")
        return False

    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path:
        print("nvidia-smi not found on PATH.")
        return False

    try:
        subprocess.run([nvidia_smi_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("NVIDIA GPU appears available.")
        return True
    except subprocess.CalledProcessError:
        print("nvidia-smi returned non-zero exit code — GPU may be unreachable or driver issue.")
        return False
    except Exception as e:
        print(f"Error checking nvidia-smi: {e}")
        return False

def run_setup():
    print("Starting GPU / rendering environment setup...")
    if False and INTEL_GPU:
        setup_intel_gpu()
        return
    check_gpu_connection()
    setup_gpu()
    add_ICD_config()

    if platform.system() == "Linux":
        mujoco_EGL_rendering()

    setup_XLA_Triton()
    print(f"Finished setup. MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    
    print("MUJOCO_GL is now:", os.environ.get("MUJOCO_GL"))
