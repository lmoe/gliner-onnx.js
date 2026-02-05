export const CONFIG_FILE = 'config.json';
export const GLINER2_CONFIG_FILE = 'gliner2_config.json';

export type Precision = 'fp32' | 'fp16';

export const TOKEN_P = '[P]';
export const TOKEN_L = '[L]';
export const TOKEN_E = '[E]';
export const TOKEN_SEP_TEXT = '[SEP_TEXT]';
export const REQUIRED_SPECIAL_TOKENS = [TOKEN_P, TOKEN_L, TOKEN_E, TOKEN_SEP_TEXT] as const;

export const SCHEMA_OPEN = '(';
export const SCHEMA_CLOSE = ')';
export const NER_TASK_NAME = 'entities';
export const CLASSIFICATION_TASK_NAME = 'category';

export const DEFAULT_THRESHOLD = 0.5;

export const ONNX_INPUT_IDS = 'input_ids';
export const ONNX_ATTENTION_MASK = 'attention_mask';
export const ONNX_HIDDEN_STATE = 'hidden_state';
export const ONNX_HIDDEN_STATES = 'hidden_states';
export const ONNX_SPAN_START_IDX = 'span_start_idx';
export const ONNX_SPAN_END_IDX = 'span_end_idx';
export const ONNX_LABEL_EMBEDDINGS = 'label_embeddings';

/** Matches URLs, emails, @mentions, words, or single non-whitespace chars. */
export const WORD_PATTERN = /(?:https?:\/\/[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S/gi;

/** Creates a fresh WORD_PATTERN instance (regex 'g' flag maintains state). */
export function createWordPattern(): RegExp {
  return new RegExp(WORD_PATTERN.source, WORD_PATTERN.flags);
}
