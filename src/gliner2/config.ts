import * as fs from 'node:fs';
import * as path from 'node:path';

import { ConfigurationError } from '../errors.js';
import type { RuntimeOptions } from '../types.js';
import { CONFIG_FILE, GLINER2_CONFIG_FILE, REQUIRED_SPECIAL_TOKENS } from './constants.js';
import type { Precision } from './constants.js';

export interface OnnxModelFiles {
  encoder: string;
  classifier: string;
  span_rep: string;
  count_embed: string;
}

export interface SpecialTokens {
  '[P]': number;
  '[L]': number;
  '[E]': number;
  '[SEP_TEXT]': number;
}

export interface GLiNER2ConfigFile {
  max_width: number;
  special_tokens: SpecialTokens & Record<string, number>;
  onnx_files: Partial<Record<Precision, OnnxModelFiles>>;
}

/** Transformer model config (config.json). Only fields we need. */
export interface TransformerConfig {
  hidden_size: number;
}

/** Loads and validates GLiNER2 configuration files. */
export class GLiNER2Config {
  public readonly modelPath: string;
  public readonly hiddenSize: number;
  public readonly maxWidth: number;
  public readonly specialTokens: SpecialTokens;
  public readonly onnxFiles: Partial<Record<Precision, OnnxModelFiles>>;

  private constructor(modelPath: string, transformerConfig: TransformerConfig, gliner2Config: GLiNER2ConfigFile, options: RuntimeOptions = {}) {
    this.modelPath = modelPath;
    this.hiddenSize = transformerConfig.hidden_size;
    this.maxWidth = options.maxWidth ?? gliner2Config.max_width;
    this.specialTokens = gliner2Config.special_tokens as SpecialTokens;
    this.onnxFiles = gliner2Config.onnx_files;
  }

  /** Load configuration from a model directory. */
  public static load(modelPath: string, options: RuntimeOptions = {}): GLiNER2Config {
    const transformerConfig = GLiNER2Config.loadTransformerConfig(modelPath);
    const gliner2Config = GLiNER2Config.loadGliner2Config(modelPath);
    return new GLiNER2Config(modelPath, transformerConfig, gliner2Config, options);
  }

  private static loadTransformerConfig(modelPath: string): TransformerConfig {
    const configPath = path.join(modelPath, CONFIG_FILE);
    if (!fs.existsSync(configPath)) {
      throw new ConfigurationError(`${CONFIG_FILE} not found in ${modelPath}`);
    }

    try {
      const content = fs.readFileSync(configPath, 'utf-8');
      const config = JSON.parse(content) as Record<string, unknown>;

      if (typeof config['hidden_size'] !== 'number') {
        throw new ConfigurationError(`${CONFIG_FILE} missing hidden_size`);
      }

      return { hidden_size: config['hidden_size'] };
    } catch (e) {
      if (e instanceof ConfigurationError) {
        throw e;
      }
      throw new ConfigurationError(`Invalid ${CONFIG_FILE}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  private static loadGliner2Config(modelPath: string): GLiNER2ConfigFile {
    const configPath = path.join(modelPath, GLINER2_CONFIG_FILE);
    if (!fs.existsSync(configPath)) {
      throw new ConfigurationError(`${GLINER2_CONFIG_FILE} not found in ${modelPath}`);
    }

    try {
      const content = fs.readFileSync(configPath, 'utf-8');
      const config = JSON.parse(content) as Record<string, unknown>;

      // Validate max_width
      if (typeof config['max_width'] !== 'number') {
        throw new ConfigurationError(`${GLINER2_CONFIG_FILE} missing max_width`);
      }

      // Validate special_tokens
      if (typeof config['special_tokens'] !== 'object' || config['special_tokens'] === null) {
        throw new ConfigurationError(`${GLINER2_CONFIG_FILE} missing special_tokens`);
      }

      const specialTokens = config['special_tokens'] as Record<string, number>;
      const missingTokens = REQUIRED_SPECIAL_TOKENS.filter((token) => !(token in specialTokens));

      if (missingTokens.length > 0) {
        throw new ConfigurationError(`${GLINER2_CONFIG_FILE} missing special tokens: ${missingTokens.join(', ')}`);
      }

      if (typeof config['onnx_files'] !== 'object' || config['onnx_files'] === null) {
        throw new ConfigurationError(`${GLINER2_CONFIG_FILE} missing onnx_files`);
      }

      return {
        max_width: config['max_width'],
        special_tokens: specialTokens as SpecialTokens & Record<string, number>,
        onnx_files: config['onnx_files'] as Partial<Record<Precision, OnnxModelFiles>>,
      };
    } catch (e) {
      if (e instanceof ConfigurationError) {
        throw e;
      }
      throw new ConfigurationError(`Invalid ${GLINER2_CONFIG_FILE}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  /** Get ONNX file paths for a specific precision, validating availability. */
  public getOnnxFiles(precision: Precision): OnnxModelFiles {
    const available = Object.keys(this.onnxFiles);
    if (available.length === 0) {
      throw new ConfigurationError(`No onnx_files found in ${GLINER2_CONFIG_FILE}`);
    }

    const files = this.onnxFiles[precision];
    if (!files) {
      throw new ConfigurationError(`Precision '${precision}' not available. Available: ${available.join(', ')}`);
    }
    return files;
  }

  public getOnnxPath(precision: Precision, model: keyof OnnxModelFiles): string {
    const files = this.getOnnxFiles(precision);
    return path.join(this.modelPath, files[model]);
  }
}
