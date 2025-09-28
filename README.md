# huge-matrix-multiply-amdgpu

The code is using rocblas to do multiplication. It is possible to use only HIP for this, but with HIP, gpu kernels need to be defined by the user.

Compile with `make matmulblas ROCBLAS=1`

You need a Laptop with >64GB system memory in order to do 50000x50000 multiplications.
To use all 64GB as GTT,set
```
options amdttm pages_limit=15728640
options amdttm page_pool_size=15728640
```
in `/etc/modprobe.d` and do `sudo update-initramfs -u -k all` to increase your GTT to 64GB.
