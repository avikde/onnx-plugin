// Copyright (c) Sample EP Authors. Licensed under the MIT License.
// Test application demonstrating how to load and use the Sample EP plugin
// Compatible with ONNX Runtime 1.23+

#include <onnxruntime_c_api.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper macro to check ORT status
#define CHECK_STATUS(expr)                                              \
    do {                                                                \
        OrtStatus* status = (expr);                                     \
        if (status != nullptr) {                                        \
            const char* msg = g_ort->GetErrorMessage(status);           \
            std::cerr << "Error: " << msg << std::endl;                 \
            g_ort->ReleaseStatus(status);                               \
            return 1;                                                   \
        }                                                               \
    } while (0)

const OrtApi* g_ort = nullptr;

int main(int argc, char* argv[]) {
    // Get the ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        std::cerr << "Failed to get ORT API" << std::endl;
        return 1;
    }

    std::cout << "ONNX Runtime Version: " << OrtGetApiBase()->GetVersionString() << std::endl;
    std::cout << "ORT API Version: " << ORT_API_VERSION << std::endl;

    // Create environment
    OrtEnv* env = nullptr;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "test_sample_ep", &env));

    // Determine the plugin library path
    const char* plugin_path = "./libsample_ep.so";
    if (argc > 1) {
        plugin_path = argv[1];
    }

    std::cout << "\nRegistering plugin EP from: " << plugin_path << std::endl;

    // Register our plugin EP library
    OrtStatus* status = g_ort->RegisterExecutionProviderLibrary(env, "SampleEP", plugin_path);
    if (status != nullptr) {
        std::cerr << "RegisterExecutionProviderLibrary failed: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return 1;
    }

    std::cout << "Plugin EP registered successfully!" << std::endl;

    // Query available EP devices
    std::cout << "\nQuerying available EP devices..." << std::endl;
    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_ep_devices = 0;
    status = g_ort->GetEpDevices(env, &ep_devices, &num_ep_devices);
    if (status != nullptr) {
        std::cerr << "GetEpDevices failed: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
    } else {
        std::cout << "Found " << num_ep_devices << " EP device(s)" << std::endl;
    }

    // Unregister the plugin EP
    std::cout << "\nUnregistering plugin EP..." << std::endl;
    status = g_ort->UnregisterExecutionProviderLibrary(env, "SampleEP");
    if (status != nullptr) {
        std::cerr << "UnregisterExecutionProviderLibrary failed: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    std::cout << "Plugin EP unregistered successfully" << std::endl;

    g_ort->ReleaseEnv(env);

    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
