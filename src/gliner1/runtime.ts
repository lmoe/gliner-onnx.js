import * as fs from 'node:fs';
import * as path from 'node:path';
import { snapshotDownload } from '@huggingface/hub';
import { AutoTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-node';
import type { InferenceSession } from 'onnxruntime-node';

import { ConfigurationError, ModelNotFoundError, ValidationError } from '../errors.js';
import type { Entity } from '../types.js';
import { SpanDecoder } from './decoder.js';
import { SpanProcessor, WhitespaceTokenSplitter } from './processor.js';
import type { GLiNER1Config, GLiNER1Tokenizer, ProcessorBatch, RawSpan } from './types.js';

const DEFAULT_THRESHOLD = 0.5;
const MODEL_FILE = 'model.onnx';
const ONNX_DIR = 'onnx';
const ONNX_OUTPUT_LOGITS = 'logits';

export class GLiNER1ONNXRuntime {
  private readonly modelPath: string;
  private readonly processor: SpanProcessor;
  private readonly decoder: SpanDecoder;
  private readonly session: InferenceSession;

  private constructor(modelPath: string, config: GLiNER1Config, session: InferenceSession, tokenizer: GLiNER1Tokenizer) {
    this.modelPath = modelPath;
    this.processor = new SpanProcessor(config, tokenizer, new WhitespaceTokenSplitter());
    this.decoder = new SpanDecoder(config);
    this.session = session;
  }

  public static async create(
    modelPath: string,
    tokenizer: GLiNER1Tokenizer,
    config: GLiNER1Config = { maxWidth: 12 },
    options?: InferenceSession.SessionOptions
  ): Promise<GLiNER1ONNXRuntime> {
    let onnxPath = modelPath;
    if (fs.existsSync(modelPath) && fs.statSync(modelPath).isDirectory()) {
      onnxPath = path.join(modelPath, ONNX_DIR, MODEL_FILE);
    }

    if (!fs.existsSync(onnxPath)) {
      throw new ModelNotFoundError(`Model not found: ${onnxPath}`);
    }

    const session = await ort.InferenceSession.create(onnxPath, options);
    return new GLiNER1ONNXRuntime(modelPath, config, session, tokenizer);
  }

  public static async fromPretrained(
    modelId: string,
    config: GLiNER1Config = { maxWidth: 12 },
    options?: InferenceSession.SessionOptions
  ): Promise<GLiNER1ONNXRuntime> {
    const modelPath = await snapshotDownload({ repo: modelId });
    const hfTokenizer = await AutoTokenizer.from_pretrained(modelPath);

    const tokenizer: GLiNER1Tokenizer = {
      encode: (text: string) => {
        const result = hfTokenizer(text, { add_special_tokens: true }) as { input_ids: { tolist: () => (bigint | number)[][] | (bigint | number)[] } };
        const ids = result.input_ids.tolist();
        const flat = Array.isArray(ids[0]) ? (ids as (bigint | number)[][]).flat() : (ids as (bigint | number)[]);
        return flat.map((v) => (typeof v === 'bigint' ? Number(v) : v));
      },
      sep_token_id: GLiNER1ONNXRuntime.getSepTokenId(hfTokenizer),
    };

    return GLiNER1ONNXRuntime.create(modelPath, tokenizer, config, options);
  }

  private static getSepTokenId(tokenizer: { sep_token_id?: unknown }): number {
    const id = tokenizer.sep_token_id;
    if (typeof id === 'bigint') {
      return Number(id);
    }
    if (typeof id === 'number') {
      return id;
    }
    return 2; // Default SEP token ID
  }

  public async extractEntities(text: string, labels: readonly string[], options: { threshold?: number } = {}): Promise<Entity[]> {
    const threshold = options.threshold ?? DEFAULT_THRESHOLD;

    if (text.trim().length === 0) {
      throw new ValidationError('Text cannot be empty');
    }
    if (labels.length === 0) {
      throw new ValidationError('Labels cannot be empty');
    }

    const rawResult = await this.runInference([text], [...labels], threshold);
    return this.convertToEntities(rawResult[0] ?? []);
  }

  public async extractEntitiesBatch(texts: string[], labels: readonly string[], options: { threshold?: number } = {}): Promise<Entity[][]> {
    const threshold = options.threshold ?? DEFAULT_THRESHOLD;

    if (texts.length === 0) {
      throw new ValidationError('Texts cannot be empty');
    }
    if (labels.length === 0) {
      throw new ValidationError('Labels cannot be empty');
    }

    const rawResult = await this.runInference(texts, [...labels], threshold);
    return rawResult.map((batchSpans) => this.convertToEntities(batchSpans));
  }

  public getModelPath(): string {
    return this.modelPath;
  }

  private convertToEntities(spans: RawSpan[]): Entity[] {
    return spans.map((span) => ({
      text: span[0],
      label: span[3],
      start: span[1],
      end: span[2],
      score: span[4],
    }));
  }

  private async runInference(texts: string[], entities: string[], threshold: number): Promise<RawSpan[][]> {
    const batch = this.processor.prepareBatch(texts, entities);
    const feeds = this.buildTensors(batch);
    const results = await this.session.run(feeds);

    const logits = results[ONNX_OUTPUT_LOGITS];
    if (!logits) {
      throw new ConfigurationError(`Model output missing ${ONNX_OUTPUT_LOGITS}`);
    }

    const batchSize = batch.batchTokens.length;
    const inputLength = Math.max(...batch.textLengths);
    const batchIds = Array.from({ length: batchSize }, (_, i) => i);

    return this.decoder.decode(
      batchSize,
      inputLength,
      entities.length,
      texts,
      batchIds,
      batch.batchWordsStartIdx,
      batch.batchWordsEndIdx,
      batch.idToClass,
      logits.data as Float32Array,
      { threshold }
    );
  }

  private buildTensors(batch: ProcessorBatch): Record<string, ort.Tensor> {
    const batchSize = batch.inputsIds.length;
    const tokenCount = batch.inputsIds[0]?.length ?? 0;
    const spanCount = batch.spanIdxs[0]?.length ?? 0;

    return {
      input_ids: new ort.Tensor('int64', this.toBigIntFlat(batch.inputsIds), [batchSize, tokenCount]),
      attention_mask: new ort.Tensor('int64', this.toBigIntFlat(batch.attentionMasks), [batchSize, tokenCount]),
      words_mask: new ort.Tensor('int64', this.toBigIntFlat(batch.wordsMasks), [batchSize, tokenCount]),
      text_lengths: new ort.Tensor('int64', batch.textLengths.map(BigInt), [batchSize, 1]),
      span_idx: new ort.Tensor('int64', this.toBigIntFlat3D(batch.spanIdxs), [batchSize, spanCount, 2]),
      span_mask: new ort.Tensor('bool', batch.spanMasks.flat(), [batchSize, spanCount]),
    };
  }

  private toBigIntFlat(arrays: number[][]): bigint[] {
    return arrays.flat().map(BigInt);
  }

  private toBigIntFlat3D(arrays: number[][][]): bigint[] {
    return arrays.flat(2).map(BigInt);
  }
}
