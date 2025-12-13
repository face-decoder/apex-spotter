The main library we will use is opencv, but for complex image processing, using only the CPU would be too burdensome. Therefore, independent compilation specifically for GPU processing is necessary to speed up image processing.

In this documentation, I will describe the steps I have taken to set up GPU-based processing.

**Disclaimer**:

> The operating system I use is [`Ubuntu 25.04`](https://releases.ubuntu.com/plucky/), so for this setup I focused on that operating system.


Follow the regular setup first from [Regular Setup](./SETUP.md)

After you have completed the initial setup via [`SETUP.md`](./SETUP.md), the next step is to prepare your CV by downloading it from packages [here](./packages/cv2-gpu.so).


Before using `cv2-gpu.so`, ensure your system has compatible NVIDIA drivers and the CUDA toolkit. OpenCV with CUDA requires an NVIDIA GPU (minimum Compute Capability 3.0).

If you don't have the Nvidia driver and toolkit yet, you can do so by following these steps

#### 1. Update system

```bash
sudo apt update && sudo apt upgrade -y
```

#### 2. Install NVIDIA drivers 

```bash
sudo apt install nvidia-driver-550 # Replace with the new driver version if any
```

#### 3. Reboot system

```bash
sudo reboot
```

#### 4. Verify driver installation

```bash
nvidia-smi
```

The command should display the name of the graphics card you are using (e.g., "NVIDIA GeForce RTX 5050").

#### 5. Install toolkit

```bash
sudo apt install nvidia-cuda-toolkit
```

Verify CUDA with running

```bash
nvcc --version
```

> <span style="color: orange">Output should be showing CUDA version like "release 12.4"</span>

> Note: This may take up around 6 GB of total storage space.


After successfully setting up the driver and toolkit, the next step is to move the opencv package that you downloaded earlier to the virtual environment that we prepared when following `SETUP.md`.

```bash
mv path-package-opencv path-project/.venv/lib/python-3.10/site-packages
```