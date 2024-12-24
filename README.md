## How to Build
For Linux Fedora 40

Install the dependencies:
```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake ninja-build libxkbcommon-devel
sudo dnf install qt6-qtbase-devel qt6-qtmultimedia-devel
sudo dnf install qt6-qtcharts-devel
sudo dnf install opencv-devel
```
Generate the Makefile and Build:

```bash
qmake && make
```

Run:
```bash
./main <grayscale_image> <reference_image> <window_size>
```

## How to set up in vscode
* Press `Ctrl+Shift+P` and search for `C/C++ Edit Configurations (JSON)`.
* Go to the .vscode folder that was created.
* Open `c_cpp_properties.json`.
* Add the OpenCV directory:
    * `"/usr/include/opencv4/**"`