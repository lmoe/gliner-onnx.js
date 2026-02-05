/** Base error for GLiNER ONNX runtime. */
export class GLiNER2Error extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GLiNER2Error';
  }
}

/** Model file not found. */
export class ModelNotFoundError extends GLiNER2Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelNotFoundError';
  }
}

/** Invalid or missing configuration. */
export class ConfigurationError extends GLiNER2Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigurationError';
  }
}

/** Input validation failed. */
export class ValidationError extends GLiNER2Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}
