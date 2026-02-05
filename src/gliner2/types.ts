import type { InferenceSession } from 'onnxruntime-node';

export type OnnxSession = InferenceSession;

export type WordOffset = [start: number, end: number];

export type Tokenizer = (
  text: string,
  options?: { add_special_tokens?: boolean }
) => Promise<{ input_ids: { tolist: () => (bigint | number)[][] | (bigint | number)[] } }>;

export type ClassificationResult = Record<string, number>;

export interface ClassifyOptions {
  /** Confidence threshold. @default 0.5 */
  threshold?: number;
  /** Allow multiple labels per input. @default false */
  multiLabel?: boolean;
}
