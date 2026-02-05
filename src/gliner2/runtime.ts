import * as fs from 'node:fs';
import { snapshotDownload } from '@huggingface/hub';
import { AutoTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-node';

import { ModelNotFoundError, ValidationError } from '../errors.js';
import type { Entity, ExtractOptions, RuntimeOptions } from '../types.js';
import { GLiNER2Config } from './config.js';
import type { SpecialTokens } from './config.js';
import {
  CLASSIFICATION_TASK_NAME,
  createWordPattern,
  DEFAULT_THRESHOLD,
  NER_TASK_NAME,
  ONNX_ATTENTION_MASK,
  ONNX_HIDDEN_STATE,
  ONNX_HIDDEN_STATES,
  ONNX_INPUT_IDS,
  ONNX_LABEL_EMBEDDINGS,
  ONNX_SPAN_END_IDX,
  ONNX_SPAN_START_IDX,
  SCHEMA_CLOSE,
  SCHEMA_OPEN,
  TOKEN_E,
  TOKEN_L,
} from './constants.js';
import type { Precision } from './constants.js';
import { computeDotProductScores, decodeEntities, decodeMultiLabel, decodeSingleLabel, generateSpans } from './decoder.js';
import type { ClassificationResult, ClassifyOptions, OnnxSession, Tokenizer, WordOffset } from './types.js';
import { copyEmbeddings, extractTokenIds, getFirstOutput, sliceHiddenStates } from './utils.js';

export class GLiNER2ONNXRuntime {
  private readonly config: GLiNER2Config;
  private readonly specialTokens: SpecialTokens;

  private readonly encoder: OnnxSession;
  private readonly classifier: OnnxSession;
  private readonly spanRep: OnnxSession;
  private readonly countEmbed: OnnxSession;

  private readonly tokenizer: Tokenizer;

  private constructor(
    config: GLiNER2Config,
    encoder: OnnxSession,
    classifier: OnnxSession,
    spanRep: OnnxSession,
    countEmbed: OnnxSession,
    tokenizer: Tokenizer
  ) {
    this.config = config;
    this.specialTokens = config.specialTokens;
    this.encoder = encoder;
    this.classifier = classifier;
    this.spanRep = spanRep;
    this.countEmbed = countEmbed;
    this.tokenizer = tokenizer;
  }

  // -------------------------------------------------------------------------
  // Factory methods
  // -------------------------------------------------------------------------

  public static async create(modelPath: string, tokenizer: Tokenizer, options: RuntimeOptions = {}): Promise<GLiNER2ONNXRuntime> {
    const providers = options.executionProviders ?? ['cpu'];
    const precision: Precision = options.precision ?? 'fp32';

    if (!fs.existsSync(modelPath) || !fs.statSync(modelPath).isDirectory()) {
      throw new ModelNotFoundError(`Model directory not found: ${modelPath}`);
    }

    const config = GLiNER2Config.load(modelPath, options);

    const [encoder, classifier, spanRep, countEmbed] = await Promise.all([
      GLiNER2ONNXRuntime.loadModel(config.getOnnxPath(precision, 'encoder'), providers),
      GLiNER2ONNXRuntime.loadModel(config.getOnnxPath(precision, 'classifier'), providers),
      GLiNER2ONNXRuntime.loadModel(config.getOnnxPath(precision, 'span_rep'), providers),
      GLiNER2ONNXRuntime.loadModel(config.getOnnxPath(precision, 'count_embed'), providers),
    ]);

    return new GLiNER2ONNXRuntime(config, encoder, classifier, spanRep, countEmbed, tokenizer);
  }

  public static async fromPretrained(modelId: string, options: RuntimeOptions = {}): Promise<GLiNER2ONNXRuntime> {
    const modelPath = await snapshotDownload({
      repo: modelId,
      accessToken: process.env['HF_TOKEN'],
    });
    const tokenizer = await AutoTokenizer.from_pretrained(modelPath);
    const tokenizerFn: Tokenizer = (text, opts) => tokenizer(text, opts) as ReturnType<Tokenizer>;
    return GLiNER2ONNXRuntime.create(modelPath, tokenizerFn, options);
  }

  private static async loadModel(filePath: string, providers: ort.InferenceSession.ExecutionProviderConfig[]): Promise<OnnxSession> {
    if (!fs.existsSync(filePath)) {
      throw new ModelNotFoundError(`Model not found: ${filePath}`);
    }
    return ort.InferenceSession.create(filePath, { executionProviders: providers });
  }

  public async classify(text: string, labels: readonly string[], options: ClassifyOptions = {}): Promise<ClassificationResult> {
    const threshold = options.threshold ?? DEFAULT_THRESHOLD;
    const multiLabel = options.multiLabel ?? false;

    this.validateInput(text, labels);

    const { tokens, labelPositions } = await this.buildSchemaInput(CLASSIFICATION_TASK_NAME, labels, TOKEN_L);
    const textTokens = await this.tokenizeText(text);
    const allTokens = [...tokens, ...textTokens];

    const hiddenStates = await this.encode(allTokens);
    const labelEmbeddings = copyEmbeddings(hiddenStates, labelPositions, this.config.hiddenSize);
    const logits = await this.runClassifier(labelEmbeddings, labels.length);

    return multiLabel ? decodeMultiLabel(logits, labels, threshold) : decodeSingleLabel(logits, labels);
  }

  public async extractEntities(text: string, labels: readonly string[], options: ExtractOptions = {}): Promise<Entity[]> {
    const threshold = options.threshold ?? DEFAULT_THRESHOLD;

    this.validateInput(text, labels);

    const { tokens: schemaTokens, labelPositions } = await this.buildSchemaInput(NER_TASK_NAME, labels, TOKEN_E);
    const textStartIdx = schemaTokens.length;
    const { tokens: textTokens, wordOffsets, firstTokenPositions } = await this.tokenizeWords(text);
    const allTokens = [...schemaTokens, ...textTokens];

    const hiddenStates = await this.encode(allTokens);
    const labelEmbeddings = copyEmbeddings(hiddenStates, labelPositions, this.config.hiddenSize);

    const textTokenCount = allTokens.length - textStartIdx;
    const wordCount = wordOffsets.length;
    if (textTokenCount === 0 || wordCount === 0) {
      return [];
    }

    const textHidden = sliceHiddenStates(hiddenStates, textStartIdx, textTokenCount, this.config.hiddenSize);
    const scores = await this.computeNerScores(textHidden, textTokenCount, wordCount, firstTokenPositions, labelEmbeddings, labels.length);

    return decodeEntities(scores, wordCount, labels, wordOffsets, text, threshold);
  }

  public getModelPath(): string {
    return this.config.modelPath;
  }

  private validateInput(text: string, labels: readonly string[]): void {
    if (text.trim().length === 0) {
      throw new ValidationError('Text cannot be empty');
    }
    if (labels.length === 0) {
      throw new ValidationError('Labels cannot be empty');
    }
  }

  private async buildSchemaInput(
    taskName: string,
    labels: readonly string[],
    labelTokenKey: '[L]' | '[E]'
  ): Promise<{ tokens: number[]; labelPositions: number[] }> {
    const pId = this.specialTokens['[P]'];
    const labelTokenId = this.specialTokens[labelTokenKey];
    const sepTextId = this.specialTokens['[SEP_TEXT]'];

    const tokens: number[] = [];

    const openParen = await this.tokenizer(SCHEMA_OPEN, { add_special_tokens: false });
    const openTokens = extractTokenIds(openParen.input_ids.tolist());
    tokens.push(...openTokens);
    tokens.push(pId);
    const taskTokens = await this.tokenizer(taskName, { add_special_tokens: false });
    tokens.push(...extractTokenIds(taskTokens.input_ids.tolist()));
    tokens.push(...openTokens);

    const labelPositions: number[] = [];
    for (const label of labels) {
      labelPositions.push(tokens.length);
      tokens.push(labelTokenId);
      const labelEncoded = await this.tokenizer(label, { add_special_tokens: false });
      tokens.push(...extractTokenIds(labelEncoded.input_ids.tolist()));
    }

    const closeParen = await this.tokenizer(SCHEMA_CLOSE, { add_special_tokens: false });
    const closeTokens = extractTokenIds(closeParen.input_ids.tolist());
    tokens.push(...closeTokens, ...closeTokens);
    tokens.push(sepTextId);

    return { tokens, labelPositions };
  }

  private async tokenizeText(text: string): Promise<number[]> {
    const tokens: number[] = [];
    const wordPattern = createWordPattern();
    const lowerText = text.toLowerCase();

    let match: RegExpExecArray | null;
    while ((match = wordPattern.exec(lowerText)) !== null) {
      const wordEncoded = await this.tokenizer(match[0], { add_special_tokens: false });
      tokens.push(...extractTokenIds(wordEncoded.input_ids.tolist()));
    }

    return tokens;
  }

  private async tokenizeWords(text: string): Promise<{ tokens: number[]; wordOffsets: WordOffset[]; firstTokenPositions: number[] }> {
    const tokens: number[] = [];
    const wordOffsets: WordOffset[] = [];
    const firstTokenPositions: number[] = [];
    const wordPattern = createWordPattern();
    const lowerText = text.toLowerCase();
    let tokenIdx = 0;

    let match: RegExpExecArray | null;
    while ((match = wordPattern.exec(lowerText)) !== null) {
      wordOffsets.push([match.index, match.index + match[0].length]);
      firstTokenPositions.push(tokenIdx);

      const wordEncoded = await this.tokenizer(match[0], { add_special_tokens: false });
      const wordTokens = extractTokenIds(wordEncoded.input_ids.tolist());
      tokens.push(...wordTokens);
      tokenIdx += wordTokens.length;
    }

    return { tokens, wordOffsets, firstTokenPositions };
  }

  private async encode(tokens: number[]): Promise<Float32Array> {
    const inputIds = new BigInt64Array(tokens.map((t) => BigInt(t)));
    const attentionMask = new BigInt64Array(tokens.length).fill(1n);
    const seqLen = tokens.length;

    const result = await this.encoder.run({
      [ONNX_INPUT_IDS]: new ort.Tensor('int64', inputIds, [1, seqLen]),
      [ONNX_ATTENTION_MASK]: new ort.Tensor('int64', attentionMask, [1, seqLen]),
    });

    return getFirstOutput(result);
  }

  private async runClassifier(labelEmbeddings: Float32Array, labelCount: number): Promise<Float32Array> {
    const result = await this.classifier.run({
      [ONNX_HIDDEN_STATE]: new ort.Tensor('float32', labelEmbeddings, [labelCount, this.config.hiddenSize]),
    });

    return getFirstOutput(result);
  }

  private async getSpanRepresentations(
    hiddenStates: Float32Array,
    seqLen: number,
    spanStart: number[],
    spanEnd: number[],
    spanCount: number
  ): Promise<Float32Array> {
    const result = await this.spanRep.run({
      [ONNX_HIDDEN_STATES]: new ort.Tensor('float32', hiddenStates, [1, seqLen, this.config.hiddenSize]),
      [ONNX_SPAN_START_IDX]: new ort.Tensor('int64', new BigInt64Array(spanStart.map(BigInt)), [1, spanCount]),
      [ONNX_SPAN_END_IDX]: new ort.Tensor('int64', new BigInt64Array(spanEnd.map(BigInt)), [1, spanCount]),
    });

    return getFirstOutput(result);
  }

  private async transformLabelEmbeddings(labelEmbeddings: Float32Array, labelCount: number): Promise<Float32Array> {
    const result = await this.countEmbed.run({
      [ONNX_LABEL_EMBEDDINGS]: new ort.Tensor('float32', labelEmbeddings, [labelCount, this.config.hiddenSize]),
    });

    return getFirstOutput(result);
  }

  private async computeNerScores(
    textHidden: Float32Array,
    textTokenCount: number,
    wordCount: number,
    firstTokenPositions: number[],
    labelEmbeddings: Float32Array,
    labelCount: number
  ): Promise<{ scores: Float32Array; wordSpanStart: number[]; wordSpanEnd: number[]; spanCount: number }> {
    const { spanStart: wordSpanStart, spanEnd: wordSpanEnd, numSpans: spanCount } = generateSpans(wordCount, this.config.maxWidth);

    const tokenSpanStart = wordSpanStart.map((wordIdx) => firstTokenPositions[wordIdx] ?? 0);
    const tokenSpanEnd = wordSpanEnd.map((wordIdx) => firstTokenPositions[wordIdx] ?? 0);

    const spanRepresentations = await this.getSpanRepresentations(textHidden, textTokenCount, tokenSpanStart, tokenSpanEnd, spanCount);
    const transformedLabels = await this.transformLabelEmbeddings(labelEmbeddings, labelCount);
    const scores = computeDotProductScores(spanRepresentations, transformedLabels, spanCount, labelCount, this.config.hiddenSize);

    return { scores, wordSpanStart, wordSpanEnd, spanCount };
  }
}
