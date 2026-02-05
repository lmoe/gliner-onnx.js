import type { InferenceSession } from 'onnxruntime-node';

import type { Precision } from './gliner2/constants.js';

export interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
  score: number;
}

export interface ExtractOptions {
  /** Confidence threshold. @default 0.5 */
  threshold?: number;
}

export interface RuntimeOptions {
  /** ONNX execution providers. @default ['cpu'] */
  executionProviders?: InferenceSession.ExecutionProviderConfig[];
  /** Model precision. @default 'fp32' */
  precision?: Precision;
  /** Maximum span width (overrides model config). */
  maxWidth?: number;
}
