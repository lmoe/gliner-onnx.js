import type { InferenceSession, Tensor } from 'onnxruntime-node';

export interface GLiNER1Config {
  maxWidth: number;
}

export type RawSpan = [text: string, start: number, end: number, label: string, score: number];

export type RawInferenceResult = RawSpan[][];

export interface ProcessorBatch {
  inputsIds: number[][];
  attentionMasks: number[][];
  wordsMasks: number[][];
  textLengths: number[];
  spanIdxs: number[][][];
  spanMasks: boolean[][];
  idToClass: Record<number, string>;
  batchTokens: string[][];
  batchWordsStartIdx: number[][];
  batchWordsEndIdx: number[][];
}

export type OrtSession = InferenceSession;

export type TensorType = 'int64' | 'bool';

export type CreateTensorFn = (type: TensorType, data: bigint[] | boolean[], dims: number[]) => Tensor;

export interface GLiNER1Tokenizer {
  encode(text: string): number[];
  sep_token_id: number;
}
