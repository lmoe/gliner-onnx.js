import type { GLiNER1Config, RawInferenceResult, RawSpan } from './types.js';

const DEFAULT_THRESHOLD = 0.5;
const ENTITY_ID_OFFSET = 1;

function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const expX = Math.exp(x);
  return expX / (1 + expX);
}

function spansOverlap(start1: number, end1: number, start2: number, end2: number, allowNested: boolean, allowMultiLabel: boolean): boolean {
  if (start1 === start2 && end1 === end2) {
    return !allowMultiLabel;
  }
  if (start1 > end2 || start2 > end1) {
    return false;
  }
  if (allowNested) {
    const isNested = (start1 <= start2 && end1 >= end2) || (start2 <= start1 && end2 >= end1);
    if (isNested) {
      return false;
    }
  }
  return true;
}

function greedySearch(spans: RawSpan[], flatNer: boolean, multiLabel: boolean): RawSpan[] {
  const sorted = spans.slice().sort((a, b) => b[4] - a[4]);
  const selected: RawSpan[] = [];

  for (const span of sorted) {
    const hasOverlap = selected.some((existing) => spansOverlap(span[1], span[2], existing[1], existing[2], !flatNer, multiLabel));
    if (!hasOverlap) {
      selected.push(span);
    }
  }

  return selected.sort((a, b) => a[1] - b[1]);
}

export interface DecoderOptions {
  /** Confidence threshold. @default 0.5 */
  threshold?: number;
  /** If true, no overlapping spans (standard NER). @default true */
  flatNer?: boolean;
  /** If true, same span can have multiple labels. @default false */
  multiLabel?: boolean;
}

/** Decoder for span-enumeration models. Output shape: [batch, seq_len, max_width, num_entities] */
export class SpanDecoder {
  private readonly maxWidth: number;

  constructor(config: GLiNER1Config) {
    this.maxWidth = config.maxWidth;
  }

  public decode(
    batchSize: number,
    inputLength: number,
    entityCount: number,
    texts: string[],
    batchIds: number[],
    charStartOffsets: number[][],
    charEndOffsets: number[][],
    idToClass: Record<number, string>,
    modelOutput: ArrayLike<number>,
    options: DecoderOptions = {}
  ): RawInferenceResult {
    const threshold = options.threshold ?? DEFAULT_THRESHOLD;
    const flatNer = options.flatNer ?? true;
    const multiLabel = options.multiLabel ?? false;

    const spans = this.extractSpans(batchSize, inputLength, entityCount, texts, batchIds, charStartOffsets, charEndOffsets, idToClass, modelOutput, threshold);

    return spans.map((batchSpans) => greedySearch(batchSpans, flatNer, multiLabel));
  }

  private extractSpans(
    batchSize: number,
    inputLength: number,
    entityCount: number,
    texts: string[],
    batchIds: number[],
    charStartOffsets: number[][],
    charEndOffsets: number[][],
    idToClass: Record<number, string>,
    modelOutput: ArrayLike<number>,
    threshold: number
  ): RawSpan[][] {
    const spans: RawSpan[][] = Array.from({ length: batchSize }, () => []);

    const batchStride = inputLength * this.maxWidth * entityCount;
    const tokenStride = this.maxWidth * entityCount;
    const widthStride = entityCount;

    for (let idx = 0; idx < modelOutput.length; idx++) {
      const prob = sigmoid(modelOutput[idx]!);
      if (prob < threshold) {
        continue;
      }

      const batchIdx = Math.floor(idx / batchStride);
      const startToken = Math.floor(idx / tokenStride) % inputLength;
      const endToken = startToken + (Math.floor(idx / widthStride) % this.maxWidth);
      const entityIdx = idx % entityCount;

      const starts = charStartOffsets[batchIdx];
      const ends = charEndOffsets[batchIdx];
      if (!starts || !ends || startToken >= starts.length || endToken >= ends.length) {
        continue;
      }

      const globalBatchIdx = batchIds[batchIdx] ?? 0;
      const text = texts[globalBatchIdx] ?? '';
      const startChar = starts[startToken]!;
      const endChar = ends[endToken]!;

      spans[batchIdx]!.push([text.slice(startChar, endChar), startChar, endChar, idToClass[entityIdx + ENTITY_ID_OFFSET] ?? '', prob]);
    }

    return spans;
  }
}
