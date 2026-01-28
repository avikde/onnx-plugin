# ONNX Runtime Execution Provider Plugin

A sample implementation of an ONNX Runtime Execution Provider (EP) plugin that can be loaded dynamically at runtime.

## Overview

This repository demonstrates how to create a custom Execution Provider plugin for ONNX Runtime 1.22+. The plugin:

- Exports the required `CreateEpFactories` and `ReleaseEpFactory` C functions
- Implements `OrtEpFactory` to create EP instances and advertise supported devices
- Implements `OrtEp` to handle node capability detection and kernel compilation
- Implements `OrtNodeComputeInfo` with `CreateState`, `Compute`, and `ReleaseState` callbacks
- Supports `Add` and `Mul` operators as a demonstration

## Prerequisites

- WSL2 (Ubuntu 22.04 or later recommended)
- CMake 3.18+
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- ONNX Runtime 1.22+ (with EP Plugin API support)

## Installing ONNX Runtime on WSL

### Install from Pre-built Release

```bash
# Create installation directory
sudo mkdir -p /opt/onnxruntime

# Download the latest release (check https://github.com/microsoft/onnxruntime/releases)
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz

# Extract to /opt/onnxruntime
sudo tar -xzf onnxruntime-linux-x64-1.23.2.tgz -C /opt/onnxruntime --strip-components=1

# Verify installation
ls /opt/onnxruntime/include
ls /opt/onnxruntime/lib
```

### Configure Library Path

Add the ONNX Runtime library to your library path:

```bash
# Add to your ~/.bashrc or ~/.zshrc
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Building the Plugin

### Clone and Build

```bash
# Clone this repository
git clone <repository-url> onnx-plugin
cd onnx-plugin

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime

# Build
cmake --build . --parallel
```

### Build Output

After building, you'll have:

- `libsample_ep.so` - The plugin EP shared library
- `test_sample_ep` - A test application (if `BUILD_TEST_APP=ON`)

### Verify Exported Symbols

```bash
nm -D libsample_ep.so | grep -E "CreateEp|ReleaseEp"
# Should show:
# T CreateEpFactories
# T ReleaseEpFactory
```

## Project Structure

```
onnx-plugin/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── include/
│   └── sample_ep.h          # EP header with class definitions
├── src/
│   └── sample_ep.cpp        # EP implementation
└── test/
    └── test_sample_ep.cpp   # Test application
```

## Plugin Architecture

### Entry Points

The plugin exports two required C functions:

```cpp
// Called by ORT to create EP factories
OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories);

// Called by ORT to release a factory
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

### OrtEpFactory

The factory creates EP instances and reports supported devices. Required callbacks:

| Callback | Purpose |
|----------|---------|
| `GetName()` | Returns EP name (e.g., "SamplePluginExecutionProvider") |
| `GetVendor()` | Returns vendor name |
| `GetVendorId()` | Returns PCI vendor ID or equivalent (1.23+) |
| `GetVersion()` | Returns semantic version string (1.23+) |
| `GetSupportedDevices()` | Enumerates hardware devices the EP can use |
| `CreateEp()` | Creates an EP instance for a session |
| `ReleaseEp()` | Releases an EP instance |

### OrtEp

The EP instance handles model execution. Required callbacks:

| Callback | Purpose |
|----------|---------|
| `GetName()` | Returns EP name (must match factory) |
| `GetCapability()` | Reports which nodes the EP can handle |
| `Compile()` | Compiles supported nodes into executable kernels |
| `ReleaseNodeComputeInfos()` | Cleans up compiled kernel info |

### OrtNodeComputeInfo

Provides the compute kernel for fused nodes:

| Callback | Purpose |
|----------|---------|
| `CreateState()` | Creates per-invocation state for the kernel |
| `Compute()` | Executes the kernel computation |
| `ReleaseState()` | Releases the state created by CreateState |

## Key Implementation Details

### Composition Pattern

The implementation uses composition rather than inheritance from `OrtEpFactory`/`OrtEp` structures:

```cpp
class SampleEpFactory {
    OrtEpFactory factory_;  // Embedded struct, returned to ORT
    std::string ep_name_;
    ApiPtrs apis_;

    // Static callbacks use CONTAINER_OF to get back to SampleEpFactory
    static SampleEpFactory* FromOrt(OrtEpFactory* ort_factory);
};
```

### noexcept Requirement

All callback functions must be declared `noexcept` to match ORT's `NO_EXCEPTION` specification:

```cpp
static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_) noexcept;
```

### API Access

ONNX Runtime APIs are accessed through function tables obtained at initialization:

```cpp
struct ApiPtrs {
    const OrtApi* ort_api;     // Main ONNX Runtime C API
    const OrtEpApi* ep_api;    // EP-specific API (GetEpApi())
};
```

## Extending the Plugin

### Adding Support for More Operators

Edit `SampleEp::GetCapabilityImpl()` in `src/sample_ep.cpp`:

```cpp
// Support additional operators
if (op_type && (std::strcmp(op_type, "Add") == 0 ||
               std::strcmp(op_type, "Mul") == 0 ||
               std::strcmp(op_type, "Sub") == 0 ||
               std::strcmp(op_type, "Div") == 0)) {
    supported_nodes.push_back(node);
}
```

Then implement the computation logic in `SampleNodeComputeInfo::ComputeImpl()`.

### Adding Hardware Device Support

To support actual hardware (GPU, NPU, etc.):

1. In `GetSupportedDevicesImpl()`, filter devices by `HardwareDevice_Type()` for your hardware type
2. Implement `CreateAllocator` callback for device memory allocation
3. Implement `CreateDataTransfer` for host-device data transfers
4. Dispatch computation to hardware in `ComputeImpl()`

## API Reference

- [ONNX Runtime Plugin EP Libraries Documentation](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html)
- [ONNX Runtime C API](https://onnxruntime.ai/docs/api/c/)
- [Example Plugin EP (ORT repo)](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/autoep/library)

## Troubleshooting

### "ONNX Runtime headers not found"

Ensure `ONNXRUNTIME_ROOT` points to the correct directory:

```bash
cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime
```

The directory should contain `include/onnxruntime_c_api.h`.

### "cannot open shared object file"

Add the ONNX Runtime library path:

```bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
```

### "undefined symbol" errors

Ensure your ONNX Runtime version supports the EP Plugin API (1.22+).

### Invalid conversion to 'noexcept' function pointer

All callback implementations must include `noexcept`:

```cpp
// Wrong:
static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_);

// Correct:
static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_) noexcept;
```

## Version Compatibility

The EP Plugin API evolved across ONNX Runtime versions:

| Version | Changes |
|---------|---------|
| 1.22 | Initial EP Plugin API |
| 1.23 | Added `GetVendorId()`, `GetVersion()`, `ValidateCompiledModelCompatibilityInfo()` |

## License

MIT License - see [LICENSE](LICENSE) for details.
