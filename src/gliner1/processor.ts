import type { GLiNER1Config, GLiNER1Tokenizer, ProcessorBatch } from './types.js';

const ENTITY_MARKER = '<<ENT>>';
const SEPARATOR_MARKER = '<<SEP>>';
const CLS_TOKEN_ID = 1;
const PAD_VALUE = 0;

export class WhitespaceTokenSplitter {
  private readonly pattern = /\w+(?:[-_]\w+)*|\S/g;

  public *split(text: string): Generator<[token: string, start: number, end: number]> {
    this.pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = this.pattern.exec(text)) !== null) {
      yield [match[0], match.index, this.pattern.lastIndex];
    }
  }
}

export class SpanProcessor {
  private readonly maxWidth: number;
  private readonly tokenizer: GLiNER1Tokenizer;
  private readonly splitter: WhitespaceTokenSplitter;

  constructor(config: GLiNER1Config, tokenizer: GLiNER1Tokenizer, splitter: WhitespaceTokenSplitter) {
    this.maxWidth = config.maxWidth;
    this.tokenizer = tokenizer;
    this.splitter = splitter;
  }

  public prepareBatch(texts: string[], entities: string[]): ProcessorBatch {
    const [batchTokens, batchWordsStartIdx, batchWordsEndIdx] = this.tokenizeTexts(texts);
    const idToClass = this.buildIdToClass(entities);
    const [inputTokens, textLengths, promptLengths] = this.buildInputSequences(batchTokens, entities);
    const [inputsIds, attentionMasks, wordsMasks] = this.encodeInputs(inputTokens, promptLengths);
    const { spanIdxs, spanMasks } = this.buildSpans(batchTokens);

    return {
      inputsIds: this.padArrays(inputsIds),
      attentionMasks: this.padArrays(attentionMasks),
      wordsMasks: this.padArrays(wordsMasks),
      textLengths,
      spanIdxs: this.padSpanArrays(spanIdxs),
      spanMasks: this.padArrays(spanMasks),
      idToClass,
      batchTokens,
      batchWordsStartIdx,
      batchWordsEndIdx,
    };
  }

  private tokenizeTexts(texts: string[]): [tokens: string[][], starts: number[][], ends: number[][]] {
    const batchTokens: string[][] = [];
    const batchStarts: number[][] = [];
    const batchEnds: number[][] = [];

    for (const text of texts) {
      const tokens: string[] = [];
      const starts: number[] = [];
      const ends: number[] = [];

      for (const [token, start, end] of this.splitter.split(text)) {
        tokens.push(token);
        starts.push(start);
        ends.push(end);
      }

      batchTokens.push(tokens);
      batchStarts.push(starts);
      batchEnds.push(ends);
    }

    return [batchTokens, batchStarts, batchEnds];
  }

  private buildIdToClass(entities: string[]): Record<number, string> {
    const idToClass: Record<number, string> = {};
    for (let i = 0; i < entities.length; i++) {
      idToClass[i + 1] = entities[i]!;
    }
    return idToClass;
  }

  private buildInputSequences(batchTokens: string[][], entities: string[]): [inputTexts: string[][], textLengths: number[], promptLengths: number[]] {
    const inputTexts: string[][] = [];
    const promptLengths: number[] = [];
    const textLengths: number[] = [];

    for (const tokens of batchTokens) {
      textLengths.push(tokens.length);

      const inputText: string[] = [];
      for (const entity of entities) {
        inputText.push(ENTITY_MARKER, entity);
      }
      inputText.push(SEPARATOR_MARKER);
      promptLengths.push(inputText.length);
      inputText.push(...tokens);
      inputTexts.push(inputText);
    }

    return [inputTexts, textLengths, promptLengths];
  }

  private encodeInputs(inputTexts: string[][], promptLengths: number[]): [inputIds: number[][], attentionMasks: number[][], wordsMasks: number[][]] {
    const inputsIds: number[][] = [];
    const attentionMasks: number[][] = [];
    const wordsMasks: number[][] = [];

    for (let batchIdx = 0; batchIdx < inputTexts.length; batchIdx++) {
      const promptLength = promptLengths[batchIdx]!;
      const tokens = inputTexts[batchIdx]!;

      const inputIds: number[] = [CLS_TOKEN_ID];
      const attentionMask: number[] = [1];
      const wordsMask: number[] = [PAD_VALUE];

      let wordCounter = 1;
      for (let wordIdx = 0; wordIdx < tokens.length; wordIdx++) {
        const wordTokenIds = this.tokenizer.encode(tokens[wordIdx]!).slice(1, -1);

        for (let tokenIdx = 0; tokenIdx < wordTokenIds.length; tokenIdx++) {
          inputIds.push(wordTokenIds[tokenIdx]!);
          attentionMask.push(1);

          if (wordIdx < promptLength) {
            wordsMask.push(PAD_VALUE);
          } else if (tokenIdx === 0) {
            wordsMask.push(wordCounter++);
          } else {
            wordsMask.push(PAD_VALUE);
          }
        }
      }

      inputIds.push(this.tokenizer.sep_token_id);
      attentionMask.push(1);
      wordsMask.push(PAD_VALUE);

      inputsIds.push(inputIds);
      attentionMasks.push(attentionMask);
      wordsMasks.push(wordsMask);
    }

    return [inputsIds, attentionMasks, wordsMasks];
  }

  private buildSpans(batchTokens: string[][]): { spanIdxs: number[][][]; spanMasks: boolean[][] } {
    const spanIdxs: number[][][] = [];
    const spanMasks: boolean[][] = [];

    for (const tokens of batchTokens) {
      const textLength = tokens.length;
      const spanIdx: number[][] = [];
      const spanMask: boolean[] = [];

      for (let startIdx = 0; startIdx < textLength; startIdx++) {
        for (let width = 0; width < this.maxWidth; width++) {
          const endIdx = Math.min(startIdx + width, textLength - 1);
          spanIdx.push([startIdx, endIdx]);
          spanMask.push(endIdx < textLength);
        }
      }

      spanIdxs.push(spanIdx);
      spanMasks.push(spanMask);
    }

    return { spanIdxs, spanMasks };
  }

  private padArrays<T>(arrays: T[][]): T[][] {
    const maxLength = Math.max(...arrays.map((arr) => arr.length));
    return arrays.map((arr) => {
      const padding = new Array<T>(maxLength - arr.length).fill(PAD_VALUE as T);
      return [...arr, ...padding];
    });
  }

  private padSpanArrays(arrays: number[][][]): number[][][] {
    const maxLength = Math.max(...arrays.map((arr) => arr.length));
    const spanDim = arrays[0]?.[0]?.length ?? 2;

    return arrays.map((arr) => {
      const padding = Array.from({ length: maxLength - arr.length }, () => new Array<number>(spanDim).fill(PAD_VALUE));
      return [...arr, ...padding];
    });
  }
}
