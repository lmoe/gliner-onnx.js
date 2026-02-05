import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Use native node environment for ONNX runtime native bindings
    pool: 'forks',
    // Increase timeout for model loading
    testTimeout: 120000,
    // Single thread to avoid ONNX conflicts
    poolOptions: {
      forks: {
        singleFork: true,
        // Ensure native modules are loaded correctly
        execArgv: ['--experimental-vm-modules'],
      },
    },
    // Don't isolate modules - needed for native bindings
    isolate: false,
  },
});
