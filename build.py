import subprocess
import sys
import os

def setup_msvc():
    if os.name == 'nt':  # Windows
        import subprocess
        try:
            # Run vswhere to find Visual Studio installation
            vswhere = (
                '"C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe" '
                '-latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 '
                '-property installationPath'
            )
            vs_path = subprocess.check_output(vswhere, shell=True).decode().strip()
            
            # Add Visual Studio and MSVC to PATH
            os.environ['PATH'] = ';'.join([
                os.environ['PATH'],
                os.path.join(vs_path, 'Common7', 'IDE'),
                os.path.join(vs_path, 'VC', 'Tools', 'MSVC', '14.37.32822', 'bin', 'Hostx64', 'x64')
            ])
        except:
            print("Warning: Could not set up MSVC environment")

def install_ninja():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja"])

def build_extension():
    try:
        # Try to build with ninja first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    except subprocess.CalledProcessError:
        print("Ninja build failed, falling back to default compiler...")
        # Fall back to default compiler
        os.environ["USE_NINJA"] = "0"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])

if __name__ == "__main__":
    setup_msvc()  # Set up MSVC environment first
    try:
        install_ninja()
    except:
        print("Failed to install ninja, will use default compiler")
    build_extension() 