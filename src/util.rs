use candle_core::Device;

/// Get either CUDA or Metal if compiled.
/// Recommended to put the override elsewhere
pub fn get_device() -> Device {
    let device: Device;
    #[cfg(feature = "cuda")]
    {
        device = Device::new_cuda(0).unwrap();
    }

    #[cfg(feature = "metal")]
    {
        device = Device::new_metal(0).unwrap();
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        device = Device::Cpu;
    }
    device
}
